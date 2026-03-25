/**
 * @file cnn_inference.c
 *
 * Pure-C forward pass for the small depth estimation CNN.
 * No external dependencies — compiles on both PC (NPS) and Bebop (AP).
 *
 * Architecture (from CreateDepthMap.py):
 *   conv1:   Conv2d(3→8,   3x3, stride=1, pad=1) + ReLU  → cross1
 *   conv2:   Conv2d(8→16,  3x3, stride=1, pad=1) + ReLU  → cross2
 *   conv3:   Conv2d(16→32, 3x3, stride=2, pad=1) + ReLU
 *   deconv1: ConvTranspose2d(32→16, 4x4, stride=2, pad=1) + ReLU
 *   cat(deconv1_out, cross2) → 32 channels
 *   deconv2: ConvTranspose2d(32→8,  3x3, stride=1, pad=1) + ReLU
 *   cat(deconv2_out, cross1) → 16 channels
 *   deconv3: ConvTranspose2d(16→1,  3x3, stride=1, pad=1) + ReLU → output
 *
 * Total parameters: 16,697
 * All buffers are static — no malloc, no stack blowup.
 */

#include "cnn_inference.h"
#include "cnn_weights.h"
#include <string.h>

/* =========================================== */
/*    Static intermediate buffers (NCHW)       */
/* =========================================== */

/* Encoder */
static float s_cross1[8  * CNN_H * CNN_W];          /* 332,800 floats */
static float s_cross2[16 * CNN_H * CNN_W];          /* 665,600 floats */
static float s_enc3  [32 * CNN_HALF_H * CNN_HALF_W]; /* 332,800 floats */

/* Decoder */
static float s_dec1  [16 * CNN_H * CNN_W];          /* 665,600 floats */
static float s_cat1  [32 * CNN_H * CNN_W];          /* 1,331,200 floats */
static float s_dec2  [8  * CNN_H * CNN_W];          /* 332,800 floats */
static float s_cat2  [16 * CNN_H * CNN_W];          /* 665,600 floats */

/* =========================================== */
/*    Conv2d: weight[out_c][in_c][kh][kw]      */
/* =========================================== */

static void conv2d(const float *input, float *output,
                   const float *weight, const float *bias,
                   int in_c, int out_c,
                   int in_h, int in_w,
                   int kh, int kw,
                   int stride, int pad)
{
  int out_h = (in_h + 2 * pad - kh) / stride + 1;
  int out_w = (in_w + 2 * pad - kw) / stride + 1;

  for (int oc = 0; oc < out_c; oc++) {
    for (int oh = 0; oh < out_h; oh++) {
      for (int ow = 0; ow < out_w; ow++) {
        float sum = bias[oc];
        for (int ic = 0; ic < in_c; ic++) {
          for (int fh = 0; fh < kh; fh++) {
            int ih = oh * stride - pad + fh;
            if (ih < 0 || ih >= in_h) continue;
            for (int fw = 0; fw < kw; fw++) {
              int iw = ow * stride - pad + fw;
              if (iw < 0 || iw >= in_w) continue;
              int w_idx = ((oc * in_c + ic) * kh + fh) * kw + fw;
              int i_idx = (ic * in_h + ih) * in_w + iw;
              sum += weight[w_idx] * input[i_idx];
            }
          }
        }
        output[(oc * out_h + oh) * out_w + ow] = sum;
      }
    }
  }
}

/* =========================================== */
/*    ConvTranspose2d: weight[in_c][out_c][kh][kw] */
/*    (PyTorch stores transposed conv this way)    */
/* =========================================== */

static void conv_transpose2d(const float *input, float *output,
                             const float *weight, const float *bias,
                             int in_c, int out_c,
                             int in_h, int in_w,
                             int kh, int kw,
                             int stride, int pad)
{
  int out_h = (in_h - 1) * stride - 2 * pad + kh;
  int out_w = (in_w - 1) * stride - 2 * pad + kw;

  /* Initialise output with bias */
  for (int oc = 0; oc < out_c; oc++) {
    float b = bias[oc];
    int base = oc * out_h * out_w;
    for (int i = 0; i < out_h * out_w; i++) {
      output[base + i] = b;
    }
  }

  /* Accumulate: iterate over input pixels (more cache-friendly for strided case) */
  for (int ic = 0; ic < in_c; ic++) {
    for (int ih = 0; ih < in_h; ih++) {
      for (int iw = 0; iw < in_w; iw++) {
        float val = input[(ic * in_h + ih) * in_w + iw];
        if (val == 0.0f) continue;  /* skip zeros from ReLU — saves ~50% work */
        for (int oc = 0; oc < out_c; oc++) {
          int w_base = ((ic * out_c + oc) * kh) * kw;
          for (int fh = 0; fh < kh; fh++) {
            int oh = ih * stride - pad + fh;
            if (oh < 0 || oh >= out_h) continue;
            int o_row = (oc * out_h + oh) * out_w;
            int w_row = w_base + fh * kw;
            for (int fw = 0; fw < kw; fw++) {
              int ow = iw * stride - pad + fw;
              if (ow < 0 || ow >= out_w) continue;
              output[o_row + ow] += weight[w_row + fw] * val;
            }
          }
        }
      }
    }
  }
}

/* =========================================== */
/*    ReLU in-place                            */
/* =========================================== */

static void relu_inplace(float *buf, int size)
{
  for (int i = 0; i < size; i++) {
    if (buf[i] < 0.0f) buf[i] = 0.0f;
  }
}

/* =========================================== */
/*    Channel concatenation (NCHW)             */
/* =========================================== */

static void concat_channels(const float *a, const float *b, float *out,
                            int ca, int cb, int h, int w)
{
  int hw = h * w;
  memcpy(out,          a, (size_t)ca * hw * sizeof(float));
  memcpy(out + ca * hw, b, (size_t)cb * hw * sizeof(float));
}

/* =========================================== */
/*    Forward pass                             */
/* =========================================== */

void cnn_forward(const float input[CNN_IN_C][CNN_H][CNN_W],
                 float output[CNN_H][CNN_W])
{
  /* ---- Encoder ---- */

  /* conv1: 3→8, 3x3, stride=1, pad=1 */
  conv2d((const float *)input, s_cross1,
         conv1_weight, conv1_bias,
         3, 8, CNN_H, CNN_W, 3, 3, 1, 1);
  relu_inplace(s_cross1, 8 * CNN_H * CNN_W);

  /* conv2: 8→16, 3x3, stride=1, pad=1 */
  conv2d(s_cross1, s_cross2,
         conv2_weight, conv2_bias,
         8, 16, CNN_H, CNN_W, 3, 3, 1, 1);
  relu_inplace(s_cross2, 16 * CNN_H * CNN_W);

  /* conv3: 16→32, 3x3, stride=2, pad=1 */
  conv2d(s_cross2, s_enc3,
         conv3_weight, conv3_bias,
         16, 32, CNN_H, CNN_W, 3, 3, 2, 1);
  relu_inplace(s_enc3, 32 * CNN_HALF_H * CNN_HALF_W);

  /* ---- Decoder ---- */

  /* deconv1: 32→16, 4x4, stride=2, pad=1 */
  conv_transpose2d(s_enc3, s_dec1,
                   deconv1_weight, deconv1_bias,
                   32, 16, CNN_HALF_H, CNN_HALF_W, 4, 4, 2, 1);
  relu_inplace(s_dec1, 16 * CNN_H * CNN_W);

  /* cat(deconv1_out, cross2) → 32 channels */
  concat_channels(s_dec1, s_cross2, s_cat1, 16, 16, CNN_H, CNN_W);

  /* deconv2: 32→8, 3x3, stride=1, pad=1 */
  conv_transpose2d(s_cat1, s_dec2,
                   deconv2_weight, deconv2_bias,
                   32, 8, CNN_H, CNN_W, 3, 3, 1, 1);
  relu_inplace(s_dec2, 8 * CNN_H * CNN_W);

  /* cat(deconv2_out, cross1) → 16 channels */
  concat_channels(s_dec2, s_cross1, s_cat2, 8, 8, CNN_H, CNN_W);

  /* deconv3: 16→1, 3x3, stride=1, pad=1 */
  conv_transpose2d(s_cat2, (float *)output,
                   deconv3_weight, deconv3_bias,
                   16, 1, CNN_H, CNN_W, 3, 3, 1, 1);
  relu_inplace((float *)output, CNN_H * CNN_W);
}
