/**
 * @file "modules/computer_vision/cnn_vision/cnn_vision.c"
 *
 * Runs a monocular depth-estimation CNN (ONNX) on the front camera and
 * produces a 7-element navigation vector (one avg-depth value per
 * horizontal strip).  The navigation module (cnn_avoid) reads this
 * vector directly — no ABI message is used between the two.
 *
 * Pipeline per frame:
 *   1. Crop a vertical strip from the YUV422 image (inlined, no function calls)
 *   2. Resize + normalise to [0,1] float NCHW tensor in a single pass
 *   3. Run ONNX inference  ->  depth map [MODEL_HEIGHT x MODEL_WIDTH]
 *   4. Split depth map into NUM_BLOCKS along HEIGHT axis, take average per block
 *   5. Store in cnn_vision_nav_vector[]
 *
 * Optimisations:
 *   - Async mode: CNN runs in background thread, camera stays at full FPS
 *   - Static buffers: no per-frame stack allocation (~790KB saved)
 *   - Inlined pixel access in crop loop (no function call per pixel)
 *   - Single resize+normalise pass (eliminates intermediate uint8 buffer)
 *   - HEIGHT-axis block split (matches real-world horizontal FOV)
 *   - memcpy for output tensor copy
 *   - Single-threaded ONNX to avoid thread contention
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

/* ========================== */
/*    Static work buffers     */
/* ========================== */

static CropBuffer s_crop;
static float s_cnn_input[1][CHANNELS][MODEL_HEIGHT][MODEL_WIDTH];
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
 *        Inlined pixel access -- no per-pixel function calls.
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
 *        directly into float NCHW tensor [1 x 3 x MODEL_HEIGHT x MODEL_WIDTH].
 *        Eliminates the intermediate uint8 resized buffer entirely.
 */
static void resize_and_normalize(const CropBuffer *crop,
                                 float input[1][CHANNELS][MODEL_HEIGHT][MODEL_WIDTH])
{
  for (int out_y = 0; out_y < MODEL_HEIGHT; out_y++) {
    int src_y = (out_y * CROP_H) / MODEL_HEIGHT;
    if (src_y >= CROP_H) src_y = CROP_H - 1;

    for (int out_x = 0; out_x < MODEL_WIDTH; out_x++) {
      int src_x = (out_x * CROP_W) / MODEL_WIDTH;
      if (src_x >= CROP_W) src_x = CROP_W - 1;

      const uint8_t *pixel = crop->data[src_y][src_x];
      input[0][0][out_y][out_x] = pixel[0] * (1.0f / 255.0f);
      input[0][1][out_y][out_x] = pixel[1] * (1.0f / 255.0f);
      input[0][2][out_y][out_x] = pixel[2] * (1.0f / 255.0f);
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

  /* Single thread — avoids contention with paparazzi threads */
  status = ort->SetIntraOpNumThreads(g_sess_opts, 1);
  if (status) goto fail;

  status = ort->CreateSession(g_env, "/home/bonkata/paparazzi/CVCNN/DroneCV/depth_model.onnx",
                              g_sess_opts, &g_session);
  if (status) goto fail;

  g_onnx_ready = true;
  fprintf(stderr, "cnn_vision: ONNX model loaded OK\n");
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

    /* Bulk copy — faster than per-element loop */
    memcpy(output, out_data, MODEL_HEIGHT * MODEL_WIDTH * sizeof(float));
  }

  if (input_name)    (void)ort->AllocatorFree(allocator, input_name);
  if (output_name)   (void)ort->AllocatorFree(allocator, output_name);
  if (output_tensor) ort->ReleaseValue(output_tensor);
  if (input_tensor)  ort->ReleaseValue(input_tensor);
  if (mem_info)      ort->ReleaseMemoryInfo(mem_info);
  return true;

fail:
  fprintf(stderr, "cnn_vision: ONNX run failed: %s\n", ort->GetErrorMessage(status));
  ort->ReleaseStatus(status);
  if (input_name  && allocator) (void)ort->AllocatorFree(allocator, input_name);
  if (output_name && allocator) (void)ort->AllocatorFree(allocator, output_name);
  if (output_tensor) ort->ReleaseValue(output_tensor);
  if (input_tensor)  ort->ReleaseValue(input_tensor);
  if (mem_info)      ort->ReleaseMemoryInfo(mem_info);
  return false;
}

/* ============================================= */
/*    Post-processing: avg depth per block       */
/* ============================================= */

/**
 * @brief Average depth per block along the HEIGHT axis.
 *
 * MODEL_HEIGHT (80) maps to the 520px real-world horizontal FOV
 * (camera is sideways, crop is resized).
 *
 *   Block 0 = rows 0-10  = one side of drone's horizontal view
 *   Block 6 = rows 69-79 = other side
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
  if (!cnn_vision_onnx_init_once()) return NULL;
  if (!crop_vertical_strip_yuv422(img, &s_crop)) return NULL;

  resize_and_normalize(&s_crop, s_cnn_input);

  if (!run_cnn_onnx(s_cnn_input, s_depth_map)) return NULL;

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
}

void cnn_vision_close(void)
{
  if (g_session)   { ort->ReleaseSession(g_session);               g_session   = NULL; }
  if (g_sess_opts) { ort->ReleaseSessionOptions(g_sess_opts);      g_sess_opts = NULL; }
  if (g_env)       { ort->ReleaseEnv(g_env);                       g_env       = NULL; }
  g_onnx_ready = false;
}