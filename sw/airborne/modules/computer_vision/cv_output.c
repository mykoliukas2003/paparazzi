#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <stdbool.h>
#include <string.h>

#include <onnxruntime_c_api.h>

#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/cv_output.h"
#include "modules/computer_vision/lib/vision/image.h"

#define MODEL_HEIGHT 80
#define MODEL_WIDTH 520
#define CHANNELS 3
#define NUM_BLOCKS 7

// Match training crop exactly:
// image.crop((80, 0, 160, 520))
#define CROP_X 80
#define CROP_Y 0
#define CROP_W 80
#define CROP_H 520

#ifndef CV_OUTPUT_FPS
#define CV_OUTPUT_FPS 0
#endif

#ifndef CV_OUTPUT_ASYNC_NICE
#define CV_OUTPUT_ASYNC_NICE 0
#endif

typedef struct {
  uint8_t data[CROP_H][CROP_W][CHANNELS];
} CropBuffer;

static const OrtApi *ort = NULL;
static OrtEnv *g_env = NULL;
static OrtSessionOptions *g_session_options = NULL;
static OrtSession *g_session = NULL;
static bool g_onnx_ready = false;

float cv_output_nav_vector[CV_OUTPUT_NUM_BLOCKS];
bool cv_output_nav_valid = false;

static inline void read_yuv422_uyvy_pixel(const struct image_t *img, int x, int y,
                                          uint8_t *Y, uint8_t *U, uint8_t *V)
{
  const uint8_t *buf = (const uint8_t *)img->buf;
  int x_even = x & ~1;
  int idx = 2 * (y * img->w + x_even);

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

static bool crop_vertical_strip_yuv422(const struct image_t *img, CropBuffer *crop)
{
  if (!img || !crop || !img->buf) {
    fprintf(stderr, "cv_output: null image input\n");
    return false;
  }

  if (img->type != IMAGE_YUV422) {
    fprintf(stderr, "cv_output: expected IMAGE_YUV422, got type=%d\n", img->type);
    return false;
  }

  if (img->w < CROP_X + CROP_W || img->h < CROP_Y + CROP_H) {
    fprintf(stderr,
            "cv_output: frame too small for crop, frame=%ux%u crop_end=(%d,%d)\n",
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

static void resize_crop_to_model(
    const CropBuffer *crop,
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

static void normalize_to_nchw(
    uint8_t resized[MODEL_HEIGHT][MODEL_WIDTH][CHANNELS],
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

static bool cv_output_onnx_init_once(void)
{
  if (g_onnx_ready) {
    return true;
  }

  ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!ort) {
    fprintf(stderr, "cv_output: failed to get ONNX Runtime API\n");
    return false;
  }

  {
    OrtStatus *status = NULL;

    status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bebop_depth", &g_env);
    if (status) goto fail;

    status = ort->CreateSessionOptions(&g_session_options);
    if (status) goto fail;

    status = ort->CreateSession(
    g_env,
    "/data/ftp/internal_000/depth_model.onnx",
    g_session_options,
    &g_session);
    if (status) goto fail;
  }

  g_onnx_ready = true;
  return true;

fail:
  fprintf(stderr, "cv_output: ONNX init failed: %s\n", ort->GetErrorMessage(status));
  ort->ReleaseStatus(status);
  return false;
}

static bool run_cnn_onnx(
    float input[1][CHANNELS][MODEL_HEIGHT][MODEL_WIDTH],
    float output[MODEL_HEIGHT][MODEL_WIDTH])
{
  if (!g_onnx_ready || !g_session) {
    return false;
  }

  OrtAllocator *allocator = NULL;
  OrtMemoryInfo *memory_info = NULL;
  OrtValue *input_tensor = NULL;
  OrtValue *output_tensor = NULL;
  char *input_name = NULL;
  char *output_name = NULL;
  OrtStatus *status = NULL;

  int64_t input_shape[] = {1, CHANNELS, MODEL_HEIGHT, MODEL_WIDTH};
  size_t input_count = 1 * CHANNELS * MODEL_HEIGHT * MODEL_WIDTH;

  status = ort->GetAllocatorWithDefaultOptions(&allocator);
  if (status) goto fail;

  status = ort->SessionGetInputName(g_session, 0, allocator, &input_name);
  if (status) goto fail;

  status = ort->SessionGetOutputName(g_session, 0, allocator, &output_name);
  if (status) goto fail;

  status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
  if (status) goto fail;

  status = ort->CreateTensorWithDataAsOrtValue(
      memory_info,
      (void *)input,
      input_count * sizeof(float),
      input_shape,
      4,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &input_tensor);
  if (status) goto fail;

  {
    const char *input_names[] = {input_name};
    const char *output_names[] = {output_name};

    status = ort->Run(
        g_session,
        NULL,
        input_names,
        (const OrtValue * const *)&input_tensor,
        1,
        output_names,
        1,
        &output_tensor);
    if (status) goto fail;
  }

  {
    float *out_data = NULL;
    status = ort->GetTensorMutableData(output_tensor, (void **)&out_data);
    if (status) goto fail;

    // Assumes output shape [1,1,80,520]
    for (int y = 0; y < MODEL_HEIGHT; y++) {
      for (int x = 0; x < MODEL_WIDTH; x++) {
        output[y][x] = out_data[y * MODEL_WIDTH + x];
      }
    }
  }

  if (input_name) ort->AllocatorFree(allocator, input_name);
  if (output_name) ort->AllocatorFree(allocator, output_name);
  if (output_tensor) ort->ReleaseValue(output_tensor);
  if (input_tensor) ort->ReleaseValue(input_tensor);
  if (memory_info) ort->ReleaseMemoryInfo(memory_info);
  return true;

fail:
  fprintf(stderr, "cv_output: ONNX run failed: %s\n", ort->GetErrorMessage(status));
  ort->ReleaseStatus(status);
  if (input_name && allocator) ort->AllocatorFree(allocator, input_name);
  if (output_name && allocator) ort->AllocatorFree(allocator, output_name);
  if (output_tensor) ort->ReleaseValue(output_tensor);
  if (input_tensor) ort->ReleaseValue(input_tensor);
  if (memory_info) ort->ReleaseMemoryInfo(memory_info);
  return false;
}

static void max_depth_per_block(float depth[MODEL_HEIGHT][MODEL_WIDTH],
                                float block_max[NUM_BLOCKS])
{
  int base_width = MODEL_WIDTH / NUM_BLOCKS;
  int remainder = MODEL_WIDTH % NUM_BLOCKS;
  int start_x = 0;

  for (int b = 0; b < NUM_BLOCKS; b++) {
    int this_width = base_width + (b < remainder ? 1 : 0);
    int end_x = start_x + this_width;

    block_max[b] = -FLT_MAX;

    for (int y = 0; y < MODEL_HEIGHT; y++) {
      for (int x = start_x; x < end_x; x++) {
        if (depth[y][x] > block_max[b]) {
          block_max[b] = depth[y][x];
        }
      }
    }

    start_x = end_x;
  }
}

static struct image_t *cv_output_func(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  if (!cv_output_onnx_init_once()) {
    return NULL;
  }

  CropBuffer crop;
  if (!crop_vertical_strip_yuv422(img, &crop)) {
    return NULL;
  }

  uint8_t resized[MODEL_HEIGHT][MODEL_WIDTH][CHANNELS];
  float cnn_input[1][CHANNELS][MODEL_HEIGHT][MODEL_WIDTH];
  float depth_map[MODEL_HEIGHT][MODEL_WIDTH];

  resize_crop_to_model(&crop, resized);
  normalize_to_nchw(resized, cnn_input);

  if (!run_cnn_onnx(cnn_input, depth_map)) {
    return NULL;
  }

  max_depth_per_block(depth_map, cv_output_nav_vector);
  cv_output_nav_valid = true;

  static int print_counter = 0;
  print_counter++;

  if (print_counter >= 10) {
    print_counter = 0;
    printf("cv_output nav_vector: [");
    for (int i = 0; i < NUM_BLOCKS; i++) {
      printf("%f", cv_output_nav_vector[i]);
      if (i < NUM_BLOCKS - 1) {
        printf(", ");
      }
    }
    printf("]\n");
    fflush(stdout);
  }
  return NULL;
}

void cv_output_init(void)
{
  cv_output_nav_valid = false;
  for (int i = 0; i < NUM_BLOCKS; i++) {
    cv_output_nav_vector[i] = 0.0f;
  }

  cv_add_to_device_async(&CV_OUTPUT_CAMERA,
                         cv_output_func,
                         CV_OUTPUT_ASYNC_NICE,
                         CV_OUTPUT_FPS,
                         0);
}

void cv_output_close(void)
{
  if (g_session) {
    ort->ReleaseSession(g_session);
    g_session = NULL;
  }
  if (g_session_options) {
    ort->ReleaseSessionOptions(g_session_options);
    g_session_options = NULL;
  }
  if (g_env) {
    ort->ReleaseEnv(g_env);
    g_env = NULL;
  }
  g_onnx_ready = false;
}
