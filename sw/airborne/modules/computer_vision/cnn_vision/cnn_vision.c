/**
 * @file "modules/computer_vision/cnn_vision/cnn_vision.c"
 *
 * Runs a monocular depth-estimation CNN on the front camera and produces
 * a 7-element navigation vector (one avg-depth value per horizontal strip).
 *
 * Uses pure-C inference (cnn_inference.c) — no ONNX Runtime dependency.
 * Compiles on both PC (NPS) and Bebop (AP).
 *
 * Pipeline per frame:
 *   1. Crop a vertical strip from the YUV422 image
 *   2. Resize + normalise to [0,1] float NCHW tensor in a single pass
 *   3. Run CNN forward pass (pure C)
 *   4. Split depth map into NUM_BLOCKS along HEIGHT axis, take average
 *   5. Store in cnn_vision_nav_vector[]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/cnn_vision/cnn_vision.h"
#include "modules/computer_vision/cnn_vision/cnn_inference.h"
#include "modules/computer_vision/lib/vision/image.h"

/* ===================== */
/*    Model geometry     */
/* ===================== */

#define MODEL_HEIGHT CNN_H
#define MODEL_WIDTH  CNN_W
#define CHANNELS     CNN_IN_C
#define NUM_BLOCKS   CNN_VISION_NUM_BLOCKS

/* Match training crop exactly: image.crop((80, 0, 160, 520)) */
#define CROP_X  80
#define CROP_Y  0
#define CROP_W  80
#define CROP_H  520

#ifndef CNN_VISION_FPS
#define CNN_VISION_FPS 2
#endif

#ifndef CNN_VISION_ASYNC_NICE
#define CNN_VISION_ASYNC_NICE 0
#endif

/* ===================== */
/*    Crop buffer type   */
/* ===================== */

typedef struct {
  uint8_t data[CROP_H][CROP_W][CHANNELS];
} CropBuffer;

/* ========================== */
/*    Static work buffers     */
/* ========================== */

static CropBuffer s_crop;
static float s_cnn_input[CHANNELS][MODEL_HEIGHT][MODEL_WIDTH];
static float s_depth_map[MODEL_HEIGHT][MODEL_WIDTH];

/* ===================== */
/*    Public output      */
/* ===================== */

float cnn_vision_nav_vector[CNN_VISION_NUM_BLOCKS];
bool  cnn_vision_nav_valid = false;

/* ============================================= */
/*    Image pre-processing                       */
/* ============================================= */

/**
 * @brief Crops a vertical strip from a YUV422 UYVY image.
 *        Inlined pixel access — no per-pixel function calls.
 */
static bool crop_vertical_strip_yuv422(const struct image_t *img,
                                       CropBuffer *crop)
{
  if (!img || !crop || !img->buf) return false;
  if (img->type != IMAGE_YUV422) return false;
  if (img->w < CROP_X + CROP_W || img->h < CROP_Y + CROP_H) return false;

  const uint8_t *buf = (const uint8_t *)img->buf;
  const int w = (int)img->w;

  for (int y = 0; y < CROP_H; y++) {
    const int src_y = CROP_Y + y;
    for (int x = 0; x < CROP_W; x++) {
      const int src_x = CROP_X + x;
      const int x_even = src_x & ~1;
      const int idx = 2 * (src_y * w + x_even);

      crop->data[y][x][0] = (src_x & 1) ? buf[idx + 3] : buf[idx + 1]; /* Y */
      crop->data[y][x][1] = buf[idx + 0];                                /* U */
      crop->data[y][x][2] = buf[idx + 2];                                /* V */
    }
  }
  return true;
}

/**
 * @brief Combined resize + normalise in a single pass.
 *        Nearest-neighbour resize from CropBuffer [CROP_H x CROP_W]
 *        directly into float NCHW tensor [3 x MODEL_HEIGHT x MODEL_WIDTH].
 */
static void resize_and_normalize(const CropBuffer *crop,
                                 float input[CHANNELS][MODEL_HEIGHT][MODEL_WIDTH])
{
  for (int out_y = 0; out_y < MODEL_HEIGHT; out_y++) {
    int src_y = (out_y * CROP_H) / MODEL_HEIGHT;
    if (src_y >= CROP_H) src_y = CROP_H - 1;

    for (int out_x = 0; out_x < MODEL_WIDTH; out_x++) {
      int src_x = (out_x * CROP_W) / MODEL_WIDTH;
      if (src_x >= CROP_W) src_x = CROP_W - 1;

      const uint8_t *pixel = crop->data[src_y][src_x];
      input[0][out_y][out_x] = pixel[0] * (1.0f / 255.0f);
      input[1][out_y][out_x] = pixel[1] * (1.0f / 255.0f);
      input[2][out_y][out_x] = pixel[2] * (1.0f / 255.0f);
    }
  }
}

/* ============================================= */
/*    Post-processing: avg depth per block       */
/* ============================================= */

/**
 * @brief Average depth per block along the HEIGHT axis.
 *
 * MODEL_HEIGHT (80) maps to the 520px real-world horizontal FOV
 * (camera is sideways, crop is resized).
 */
static void avg_depth_per_block(float depth[MODEL_HEIGHT][MODEL_WIDTH],
                                float block_vals[NUM_BLOCKS])
{
  int base_height = MODEL_HEIGHT / NUM_BLOCKS;
  int remainder   = MODEL_HEIGHT % NUM_BLOCKS;
  int start_y     = 0;

  for (int b = 0; b < NUM_BLOCKS; b++) {
    int this_height = base_height + (b < remainder ? 1 : 0);
    int end_y       = start_y + this_height;

    float sum = 0.0f;
    int count = 0;

    for (int y = start_y; y < end_y; y++) {
      for (int x = 0; x < MODEL_WIDTH; x++) {
        sum += depth[y][x];
        count++;
      }
    }

    block_vals[b] = (count > 0) ? sum / (float)count : 0.0f;
    start_y = end_y;
  }
}

/* ============================================= */
/*    CV callback (runs in async video thread)   */
/* ============================================= */

static struct image_t *cnn_vision_func(struct image_t *img,
                                       uint8_t camera_id __attribute__((unused)))
{
  if (!crop_vertical_strip_yuv422(img, &s_crop)) return NULL;

  resize_and_normalize(&s_crop, s_cnn_input);

  /* Run pure-C CNN forward pass — no ONNX Runtime needed */
  cnn_forward(s_cnn_input, s_depth_map);

  avg_depth_per_block(s_depth_map, cnn_vision_nav_vector);
  cnn_vision_nav_valid = true;

  /* Periodic debug print (every 20th frame) */
  static int print_counter = 0;
  if (++print_counter >= 20) {
    print_counter = 0;
    fprintf(stderr, "cnn_vision: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
            cnn_vision_nav_vector[0], cnn_vision_nav_vector[1],
            cnn_vision_nav_vector[2], cnn_vision_nav_vector[3],
            cnn_vision_nav_vector[4], cnn_vision_nav_vector[5],
            cnn_vision_nav_vector[6]);
  }

  return NULL;
}

/* ============================================= */
/*    Public init / close                        */
/* ============================================= */

void cnn_vision_init(void)
{
  cnn_vision_nav_valid = false;
  for (int i = 0; i < CNN_VISION_NUM_BLOCKS; i++) {
    cnn_vision_nav_vector[i] = 0.0f;
  }

  cv_add_to_device_async(&CNN_VISION_CAMERA,
                         cnn_vision_func,
                         CNN_VISION_ASYNC_NICE,
                         CNN_VISION_FPS,
                         0);

  fprintf(stderr, "cnn_vision: init complete (pure-C inference, %d params)\n", 16697);
}

void cnn_vision_close(void)
{
  /* Nothing to clean up — all buffers are static */
}