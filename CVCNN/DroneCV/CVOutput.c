#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <string.h>
#include <pthread.h>
#include <onnxruntime_c_api.h>

#define MODEL_HEIGHT 80
#define MODEL_WIDTH 520
#define CHANNELS 3
#define NUM_BLOCKS 7

// Training crop from Python:
// image.crop((80, 0, 160, 520))
// => x=80, y=0, width=80, height=520
#define CROP_X 80
#define CROP_Y 0
#define CROP_W 80
#define CROP_H 520

typedef enum {
    YUV_FORMAT_UNKNOWN = 0,
    YUV_FORMAT_NV12
} YuvFormat;

typedef struct {
    const uint8_t* y_plane;
    const uint8_t* uv_plane;   // NV12 only

    int width;
    int height;

    int y_stride;
    int uv_stride;

    YuvFormat format;
} YuvFrame;

typedef struct {
    uint8_t data[CROP_H][CROP_W][CHANNELS];
} CropBuffer;

// Thread-safe owned copy of the latest decoded frame
typedef struct {
    uint8_t* y_plane;
    uint8_t* uv_plane;

    int width;
    int height;

    int y_stride;
    int uv_stride;

    YuvFormat format;
    int valid;
} OwnedYuvFrame;

static pthread_mutex_t g_frame_mutex = PTHREAD_MUTEX_INITIALIZER;
static OwnedYuvFrame g_latest_frame = {0};

static const OrtApi* ort = NULL;

#define ORT_CHECK(expr)                                                        \
    do {                                                                       \
        OrtStatus* status__ = (expr);                                          \
        if (status__ != NULL) {                                                \
            fprintf(stderr, "ONNX Runtime error: %s\n",                        \
                    ort->GetErrorMessage(status__));                            \
            ort->ReleaseStatus(status__);                                      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static void free_owned_frame(OwnedYuvFrame* f) {
    if (!f) return;
    free(f->y_plane);
    free(f->uv_plane);
    memset(f, 0, sizeof(*f));
}

/*
 * Call this from your Bebop decoded-frame callback whenever a new NV12 frame arrives.
 * This function makes its own copy, so the SDK can release its buffers afterward.
 */
void bebop_update_latest_frame_nv12(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    int width,
    int height,
    int y_stride,
    int uv_stride
) {
    if (!y_plane || !uv_plane || width <= 0 || height <= 0 ||
        y_stride <= 0 || uv_stride <= 0) {
        return;
    }

    pthread_mutex_lock(&g_frame_mutex);

    free_owned_frame(&g_latest_frame);

    size_t y_bytes  = (size_t)y_stride * (size_t)height;
    size_t uv_bytes = (size_t)uv_stride * (size_t)(height / 2);

    g_latest_frame.y_plane = (uint8_t*)malloc(y_bytes);
    g_latest_frame.uv_plane = (uint8_t*)malloc(uv_bytes);

    if (!g_latest_frame.y_plane || !g_latest_frame.uv_plane) {
        free_owned_frame(&g_latest_frame);
        pthread_mutex_unlock(&g_frame_mutex);
        return;
    }

    memcpy(g_latest_frame.y_plane, y_plane, y_bytes);
    memcpy(g_latest_frame.uv_plane, uv_plane, uv_bytes);

    g_latest_frame.width = width;
    g_latest_frame.height = height;
    g_latest_frame.y_stride = y_stride;
    g_latest_frame.uv_stride = uv_stride;
    g_latest_frame.format = YUV_FORMAT_NV12;
    g_latest_frame.valid = 1;

    pthread_mutex_unlock(&g_frame_mutex);
}

/*
 * Pull the most recent frame into a lightweight view.
 * Assumes latest decoded frame is NV12.
 */
int capture_bebop_yuv_frame(YuvFrame* frame) {
    if (!frame) {
        return 0;
    }

    pthread_mutex_lock(&g_frame_mutex);

    if (!g_latest_frame.valid || g_latest_frame.format != YUV_FORMAT_NV12) {
        pthread_mutex_unlock(&g_frame_mutex);
        return 0;
    }

    frame->y_plane  = g_latest_frame.y_plane;
    frame->uv_plane = g_latest_frame.uv_plane;

    frame->width    = g_latest_frame.width;
    frame->height   = g_latest_frame.height;

    frame->y_stride  = g_latest_frame.y_stride;
    frame->uv_stride = g_latest_frame.uv_stride;

    frame->format = YUV_FORMAT_NV12;

    pthread_mutex_unlock(&g_frame_mutex);
    return 1;
}

// Keep channels as Y, U, V-like values to stay close to training on YCbCr
static inline void pack_yuv_channels(uint8_t y, uint8_t u, uint8_t v, uint8_t out[3]) {
    out[0] = y;
    out[1] = u;
    out[2] = v;
}

static inline void read_nv12_pixel(const YuvFrame* f, int x, int y,
                                   uint8_t* Y, uint8_t* U, uint8_t* V) {
    *Y = f->y_plane[y * f->y_stride + x];

    int uv_x = (x / 2) * 2;
    int uv_y = y / 2;

    const uint8_t* uv_row = f->uv_plane + uv_y * f->uv_stride;
    *U = uv_row[uv_x + 0];
    *V = uv_row[uv_x + 1];
}

// 1) Crop the same vertical strip used in training
void crop_vertical_strip_yuv(const YuvFrame* frame, CropBuffer* crop) {
    if (!frame || !crop) {
        fprintf(stderr, "crop_vertical_strip_yuv: null input\n");
        exit(EXIT_FAILURE);
    }

    if (frame->format != YUV_FORMAT_NV12) {
        fprintf(stderr, "Unsupported YUV format\n");
        exit(EXIT_FAILURE);
    }

    if (frame->width < CROP_X + CROP_W || frame->height < CROP_Y + CROP_H) {
        fprintf(stderr, "Frame too small for configured crop\n");
        fprintf(stderr, "Frame size: %dx%d, crop requires up to (%d,%d)\n",
                frame->width, frame->height, CROP_X + CROP_W, CROP_Y + CROP_H);
        exit(EXIT_FAILURE);
    }

    for (int y = 0; y < CROP_H; y++) {
        for (int x = 0; x < CROP_W; x++) {
            uint8_t Y, U, V;
            read_nv12_pixel(frame, CROP_X + x, CROP_Y + y, &Y, &U, &V);
            pack_yuv_channels(Y, U, V, crop->data[y][x]);
        }
    }
}

// 2) Resize crop (80x520 vertical strip) to model input (80x520 tensor shape)
//    using nearest-neighbor for simplicity
void resize_crop_to_model(
    const CropBuffer* crop,
    uint8_t resized[MODEL_HEIGHT][MODEL_WIDTH][CHANNELS]) {

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

// 3) Normalize to float tensor [1, 3, 80, 520], matching ToTensor() behavior
void normalize_to_nchw(
    uint8_t resized[MODEL_HEIGHT][MODEL_WIDTH][CHANNELS],
    float input[1][CHANNELS][MODEL_HEIGHT][MODEL_WIDTH]) {

    for (int c = 0; c < CHANNELS; c++) {
        for (int y = 0; y < MODEL_HEIGHT; y++) {
            for (int x = 0; x < MODEL_WIDTH; x++) {
                input[0][c][y][x] = resized[y][x][c] / 255.0f;
            }
        }
    }
}

// 4) Run ONNX model, output depth map [80][520]
void run_cnn_onnx(
    OrtSession* session,
    float input[1][CHANNELS][MODEL_HEIGHT][MODEL_WIDTH],
    float output[MODEL_HEIGHT][MODEL_WIDTH]) {

    OrtAllocator* allocator = NULL;
    OrtMemoryInfo* memory_info = NULL;
    OrtValue* input_tensor = NULL;
    OrtValue* output_tensor = NULL;
    char* input_name = NULL;
    char* output_name = NULL;

    int64_t input_shape[] = {1, CHANNELS, MODEL_HEIGHT, MODEL_WIDTH};
    size_t input_count = 1 * CHANNELS * MODEL_HEIGHT * MODEL_WIDTH;

    ORT_CHECK(ort->GetAllocatorWithDefaultOptions(&allocator));
    ORT_CHECK(ort->SessionGetInputName(session, 0, allocator, &input_name));
    ORT_CHECK(ort->SessionGetOutputName(session, 0, allocator, &output_name));

    ORT_CHECK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    ORT_CHECK(ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)input,
        input_count * sizeof(float),
        input_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor));

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};

    ORT_CHECK(ort->Run(
        session,
        NULL,
        input_names,
        (const OrtValue* const*)&input_tensor,
        1,
        output_names,
        1,
        &output_tensor));

    float* out_data = NULL;
    ORT_CHECK(ort->GetTensorMutableData(output_tensor, (void**)&out_data));

    // Assumes output shape [1,1,80,520]
    for (int y = 0; y < MODEL_HEIGHT; y++) {
        for (int x = 0; x < MODEL_WIDTH; x++) {
            output[y][x] = out_data[y * MODEL_WIDTH + x];
        }
    }

    if (input_name) ort->AllocatorFree(allocator, input_name);
    if (output_name) ort->AllocatorFree(allocator, output_name);
    if (output_tensor) ort->ReleaseValue(output_tensor);
    if (input_tensor) ort->ReleaseValue(input_tensor);
    if (memory_info) ort->ReleaseMemoryInfo(memory_info);
}

// 5) Compute 1x7 vector of max depth values, one max per block
void max_depth_per_block(float depth[MODEL_HEIGHT][MODEL_WIDTH],
                         float block_max[NUM_BLOCKS]) {
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

int main(void) {
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        return EXIT_FAILURE;
    }

    // In a real Bebop pipeline, your decoded-frame callback should have already
    // called bebop_update_latest_frame_nv12(...) before this point.

    YuvFrame frame;
    if (!capture_bebop_yuv_frame(&frame)) {
        fprintf(stderr, "Error: failed to capture Bebop NV12 frame\n");
        return EXIT_FAILURE;
    }

    CropBuffer crop;
    crop_vertical_strip_yuv(&frame, &crop);

    uint8_t resized[MODEL_HEIGHT][MODEL_WIDTH][CHANNELS];
    resize_crop_to_model(&crop, resized);

    float cnn_input[1][CHANNELS][MODEL_HEIGHT][MODEL_WIDTH];
    normalize_to_nchw(resized, cnn_input);

    OrtEnv* env = NULL;
    OrtSessionOptions* session_options = NULL;
    OrtSession* session = NULL;

    ORT_CHECK(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bebop_depth", &env));
    ORT_CHECK(ort->CreateSessionOptions(&session_options));
    ORT_CHECK(ort->CreateSession(env, "depth_model.onnx", session_options, &session));

    float depth_map[MODEL_HEIGHT][MODEL_WIDTH];
    run_cnn_onnx(session, cnn_input, depth_map);

    float nav_vector[NUM_BLOCKS];
    max_depth_per_block(depth_map, nav_vector);

    printf("Navigation vector [");
    for (int i = 0; i < NUM_BLOCKS; i++) {
        printf("%f", nav_vector[i]);
        if (i < NUM_BLOCKS - 1) {
            printf(", ");
        }
    }
    printf("]\n");

    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(session_options);
    ort->ReleaseEnv(env);

    pthread_mutex_lock(&g_frame_mutex);
    free_owned_frame(&g_latest_frame);
    pthread_mutex_unlock(&g_frame_mutex);

    return EXIT_SUCCESS;
}