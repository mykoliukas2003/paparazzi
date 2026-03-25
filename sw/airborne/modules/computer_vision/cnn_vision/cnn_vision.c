/**
 * @file "modules/computer_vision/cnn_vision/cnn_vision.c"
 *
 * Runs a monocular depth-estimation CNN (ONNX) on the front camera and
 * produces a 7-element navigation vector (one max-depth value per
 * horizontal strip).  The navigation module (cnn_avoid) reads this
 * vector directly — no ABI message is used between the two.
 *
 * Pipeline per frame:
 *   1. Crop a vertical strip from the YUV422 image
 *   2. Resize to MODEL_HEIGHT x MODEL_WIDTH
 *   3. Normalise to [0,1] float NCHW tensor
 *   4. Run ONNX inference  →  depth map [MODEL_HEIGHT x MODEL_WIDTH]
 *   5. Split depth map into NUM_BLOCKS vertical strips, take max per strip
 *   6. Store in cnn_vision_nav_vector[]
 *   7. (Optional) Draw depth overlay on the camera feed
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <stdbool.h>
#include <string.h>

#include <onnxruntime_c_api.h>

#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/cnn_vision/cnn_vision.h"
#include "modules/computer_vision/lib/vision/image.h"

/* ===================== */
/*    Model geometry     */
/* ===================== */

#define MODEL_HEIGHT 80
#define MODEL_WIDTH  520
#define CHANNELS     3
#define NUM_BLOCKS   CNN_VISION_NUM_BLOCKS

/* Match training crop exactly: image.crop((80, 0, 160, 520)) */
#define CROP_X  80
#define CROP_Y  0
#define CROP_W  80
#define CROP_H  520

#ifndef CNN_VISION_FPS
#define CNN_VISION_FPS 0
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

/* ===================== */
/*    ONNX state         */
/* ===================== */

static const OrtApi *ort              = NULL;
static OrtEnv *g_env                  = NULL;
static OrtSessionOptions *g_sess_opts = NULL;
static OrtSession *g_session          = NULL;
static bool g_onnx_ready              = false;

/* ===================== */
/*    Public output      */
/* ===================== */

float cnn_vision_nav_vector[CNN_VISION_NUM_BLOCKS];
bool  cnn_vision_nav_valid = false;

/* Draw depth overlay on camera feed — togglable from GCS */
uint8_t cnn_vision_draw = 1;

/* ============================================= */
/*    YUV422 pixel reader (UYVY byte order)      */
/* ============================================= */

static inline void read_yuv422_uyvy_pixel(const struct image_t *img,
                                          int x, int y,
                                          uint8_t *Y, uint8_t *U, uint8_t *V)
{
  const uint8_t *buf = (const uint8_t *)img->buf;
  int x_even = x & ~1;
  int idx    = 2 * (y * img->w + x_even);

  *U = buf[idx + 0];
  *V = buf[idx + 2];
  *Y = (x % 2 == 0) ? buf[idx + 1] : buf[idx + 3];
}

static inline void pack_yuv_channels(uint8_t y, uint8_t u, uint8_t v,
                                     uint8_t out[3])
{
  out[0] = y;
  out[1] = u;
  out[2] = v;
}

/* ============================================= */
/*    Image pre-processing                       */
/* ============================================= */

static bool crop_vertical_strip_yuv422(const struct image_t *img,
                                       CropBuffer *crop)
{
  if (!img || !crop || !img->buf) {
    fprintf(stderr, "cnn_vision: null image input\n");
    return false;
  }
  if (img->type != IMAGE_YUV422) {
    fprintf(stderr, "cnn_vision: expected IMAGE_YUV422, got type=%d\n", img->type);
    return false;
  }
  if (img->w < CROP_X + CROP_W || img->h < CROP_Y + CROP_H) {
    fprintf(stderr, "cnn_vision: frame too small (%ux%u), need (%d,%d)\n",
            img->w, img->h, CROP_X + CROP_W, CROP_Y + CROP_H);
    return false;
  }

  for (int y = 0; y < CROP_H; y++) {
    for (int x = 0; x < CROP_W; x++) {
      uint8_t Y, U, V;
      read_yuv422_uyvy_pixel(img, CROP_X + x, CROP_Y + y, &Y, &U, &V);
      pack_yuv_channels(Y, U, V, crop->data[y][x]);
    }
  }
  return true;
}

static void resize_crop_to_model(const CropBuffer *crop,
                                 uint8_t resized[MODEL_HEIGHT][MODEL_WIDTH][CHANNELS])
{
  for (int out_y = 0; out_y < MODEL_HEIGHT; out_y++) {
    for (int out_x = 0; out_x < MODEL_WIDTH; out_x++) {
      int src_y = (out_y * CROP_H) / MODEL_HEIGHT;
      int src_x = (out_x * CROP_W) / MODEL_WIDTH;
      if (src_y >= CROP_H) src_y = CROP_H - 1;
      if (src_x >= CROP_W) src_x = CROP_W - 1;

      for (int c = 0; c < CHANNELS; c++) {
        resized[out_y][out_x][c] = crop->data[src_y][src_x][c];
      }
    }
  }
}

static void normalize_to_nchw(uint8_t resized[MODEL_HEIGHT][MODEL_WIDTH][CHANNELS],
                               float input[1][CHANNELS][MODEL_HEIGHT][MODEL_WIDTH])
{
  for (int c = 0; c < CHANNELS; c++) {
    for (int y = 0; y < MODEL_HEIGHT; y++) {
      for (int x = 0; x < MODEL_WIDTH; x++) {
        input[0][c][y][x] = resized[y][x][c] / 255.0f;
      }
    }
  }
}

/* ============================================= */
/*    ONNX Runtime setup                         */
/* ============================================= */

static bool cnn_vision_onnx_init_once(void)
{
  if (g_onnx_ready) return true;

  ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!ort) {
    fprintf(stderr, "cnn_vision: failed to get ONNX Runtime API\n");
    return false;
  }

  OrtStatus *status = NULL;

  status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "cnn_depth", &g_env);
  if (status) goto fail;

  status = ort->CreateSessionOptions(&g_sess_opts);
  if (status) goto fail;

  status = ort->CreateSession(g_env, "/home/bonkata/paparazzi/CVCNN/DroneCV/depth_model.onnx", g_sess_opts, &g_session);
  if (status) goto fail;

  g_onnx_ready = true;
  return true;

fail:
  fprintf(stderr, "cnn_vision: ONNX init failed: %s\n", ort->GetErrorMessage(status));
  ort->ReleaseStatus(status);
  return false;
}

/* ============================================= */
/*    ONNX inference                             */
/* ============================================= */

static bool run_cnn_onnx(float input[1][CHANNELS][MODEL_HEIGHT][MODEL_WIDTH],
                         float output[MODEL_HEIGHT][MODEL_WIDTH])
{
  if (!g_onnx_ready || !g_session) return false;

  OrtAllocator *allocator  = NULL;
  OrtMemoryInfo *mem_info  = NULL;
  OrtValue *input_tensor   = NULL;
  OrtValue *output_tensor  = NULL;
  char *input_name         = NULL;
  char *output_name        = NULL;
  OrtStatus *status        = NULL;

  int64_t input_shape[] = {1, CHANNELS, MODEL_HEIGHT, MODEL_WIDTH};
  size_t  input_count   = 1 * CHANNELS * MODEL_HEIGHT * MODEL_WIDTH;

  status = ort->GetAllocatorWithDefaultOptions(&allocator);
  if (status) goto fail;
  status = ort->SessionGetInputName(g_session, 0, allocator, &input_name);
  if (status) goto fail;
  status = ort->SessionGetOutputName(g_session, 0, allocator, &output_name);
  if (status) goto fail;
  status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
  if (status) goto fail;

  status = ort->CreateTensorWithDataAsOrtValue(
      mem_info, (void *)input,
      input_count * sizeof(float),
      input_shape, 4,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &input_tensor);
  if (status) goto fail;

  {
    const char *in_names[]  = {input_name};
    const char *out_names[] = {output_name};
    status = ort->Run(g_session, NULL,
                      in_names, (const OrtValue *const *)&input_tensor, 1,
                      out_names, 1, &output_tensor);
    if (status) goto fail;
  }

  {
    float *out_data = NULL;
    status = ort->GetTensorMutableData(output_tensor, (void **)&out_data);
    if (status) goto fail;

    /* Output shape assumed [1, 1, MODEL_HEIGHT, MODEL_WIDTH] */
    for (int y = 0; y < MODEL_HEIGHT; y++) {
      for (int x = 0; x < MODEL_WIDTH; x++) {
        output[y][x] = out_data[y * MODEL_WIDTH + x];
      }
    }
  }

  if (input_name)    ort->AllocatorFree(allocator, input_name);
  if (output_name)   ort->AllocatorFree(allocator, output_name);
  if (output_tensor) ort->ReleaseValue(output_tensor);
  if (input_tensor)  ort->ReleaseValue(input_tensor);
  if (mem_info)      ort->ReleaseMemoryInfo(mem_info);
  return true;

fail:
  fprintf(stderr, "cnn_vision: ONNX run failed: %s\n", ort->GetErrorMessage(status));
  ort->ReleaseStatus(status);
  if (input_name  && allocator) ort->AllocatorFree(allocator, input_name);
  if (output_name && allocator) ort->AllocatorFree(allocator, output_name);
  if (output_tensor) ort->ReleaseValue(output_tensor);
  if (input_tensor)  ort->ReleaseValue(input_tensor);
  if (mem_info)      ort->ReleaseMemoryInfo(mem_info);
  return false;
}

/**
 * @brief Average depth per block along the HEIGHT axis.
 *        More robust than MIN (single bad pixel doesn't dominate)
 *        or MAX (hides nearby obstacles).
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
/*    Depth overlay drawing                      */
/* ============================================= */

 /* @brief Draws 7 coloured horizontal strips on the raw image.
 *        After the GStreamer counterclockwise rotation, these appear
 *        as vertical columns on screen — matching the drone's left/right view.
 *
 *        Bar length (growing from the right edge) is proportional to depth.
 *        After rotation, bars grow upward from the bottom of the screen.
 *
 *        Block order is reversed so that after CCW rotation:
 *          screen left  = block 0-1 = drone's LEFT
 *          screen centre = block 2-4 = STRAIGHT ahead
 *          screen right = block 5-6 = drone's RIGHT
 */
static void draw_depth_overlay(struct image_t *img,
                               float block_depths[CNN_VISION_NUM_BLOCKS])
{
  if (!img || !img->buf) return;

  uint8_t *buf = (uint8_t *)img->buf;
  int strip_height = (int)img->h / CNN_VISION_NUM_BLOCKS;

  for (int b = 0; b < CNN_VISION_NUM_BLOCKS; b++) {
    /* Reverse block order: bottom of raw → left of rotated screen */
    int draw_b = CNN_VISION_NUM_BLOCKS - 1 - b;
    float d = block_depths[draw_b];

    uint8_t Y_val, U_val, V_val;
    if (d < 0.48f) {
      Y_val = 76;  U_val = 84;  V_val = 255;   /* Red — obstacle */
    } else if (d < 0.57f) {
      Y_val = 226; U_val = 0;   V_val = 149;   /* Yellow — caution */
    } else {
      Y_val = 150; U_val = 43;  V_val = 21;    /* Green — safe */
    }

    /* Bar width proportional to depth, growing from the right edge */
    int bar_width = (int)(d * (float)img->w);
    if (bar_width > (int)img->w) bar_width = (int)img->w;
    int x_start = (int)img->w - bar_width;

    int y_start = b * strip_height;
    int y_end   = y_start + strip_height;
    if (y_end > (int)img->h) y_end = (int)img->h;

    for (int y = y_start; y < y_end; y++) {
      for (int x = x_start; x < (int)img->w; x++) {
        int x_even = x & ~1;
        int idx = 2 * (y * (int)img->w + x_even);

        buf[idx + 0] = (uint8_t)((buf[idx + 0] + U_val) / 2);
        buf[idx + 2] = (uint8_t)((buf[idx + 2] + V_val) / 2);
        if (x % 2 == 0) {
          buf[idx + 1] = (uint8_t)((buf[idx + 1] + Y_val) / 2);
        } else {
          buf[idx + 3] = (uint8_t)((buf[idx + 3] + Y_val) / 2);
        }
      }
    }

    /* White separator line between strips */
    if (b > 0) {
      for (int x = 0; x < (int)img->w; x++) {
        int x_even = x & ~1;
        int idx = 2 * (y_start * (int)img->w + x_even);
        if (x % 2 == 0) {
          buf[idx + 1] = 235;
        } else {
          buf[idx + 3] = 235;
        }
        buf[idx + 0] = 128;
        buf[idx + 2] = 128;
      }
    }
  }
}

/* ============================================= */
/*    CV callback (runs in video thread)         */
/* ============================================= */

static struct image_t *cnn_vision_func(struct image_t *img,
                                       uint8_t camera_id __attribute__((unused)))
{
  if (!cnn_vision_onnx_init_once()) return img;

  CropBuffer crop;
  if (!crop_vertical_strip_yuv422(img, &crop)) return img;

  uint8_t resized[MODEL_HEIGHT][MODEL_WIDTH][CHANNELS];
  float   cnn_input[1][CHANNELS][MODEL_HEIGHT][MODEL_WIDTH];
  float   depth_map[MODEL_HEIGHT][MODEL_WIDTH];

  resize_crop_to_model(&crop, resized);
  normalize_to_nchw(resized, cnn_input);

  if (!run_cnn_onnx(cnn_input, depth_map)) return img;

  avg_depth_per_block(depth_map, cnn_vision_nav_vector);
  cnn_vision_nav_valid = true;

  /* Draw depth overlay on camera feed if enabled */
  if (cnn_vision_draw) {
    draw_depth_overlay(img, cnn_vision_nav_vector);
  }

  /* Periodic debug print (every 10th frame) */
  static int print_counter = 0;
  if (++print_counter >= 10) {
    print_counter = 0;
    fprintf(stderr, "cnn_vision nav_vector: [");
    for (int i = 0; i < NUM_BLOCKS; i++) {
      fprintf(stderr, "%.2f%s", cnn_vision_nav_vector[i],
              (i < NUM_BLOCKS - 1) ? ", " : "");
    }
    fprintf(stderr, "]\n");
  }

  return img;  /* Return img so the video pipeline passes it to the RTP stream */
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

    cv_add_to_device(&CNN_VISION_CAMERA,
                    cnn_vision_func,
                    CNN_VISION_FPS,
                    0);
}

void cnn_vision_close(void)
{
  if (g_session)   { ort->ReleaseSession(g_session);               g_session   = NULL; }
  if (g_sess_opts) { ort->ReleaseSessionOptions(g_sess_opts);      g_sess_opts = NULL; }
  if (g_env)       { ort->ReleaseEnv(g_env);                       g_env       = NULL; }
  g_onnx_ready = false;
}