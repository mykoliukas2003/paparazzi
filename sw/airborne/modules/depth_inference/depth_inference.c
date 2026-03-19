#include "depth_inference.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

// stb_image_write — single header, no external dependency
// Place stb_image_write.h in paparazzi/sw/airborne/modules/depth_inference/
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ONNX Runtime
#include <onnxruntime_c_api.h>

// Paparazzi includes
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"

// ------------- CONFIG -------------
#define TARGET_WIDTH  240
#define TARGET_HEIGHT 520

// How many seconds between saved depth images (change to taste)
#ifndef DEPTH_SAVE_INTERVAL_SEC
#define DEPTH_SAVE_INTERVAL_SEC 5
#endif

// Where to save depth images
// Simulator: /tmp/paparazzi/  — always exists during sim
// Drone:     /data/ftp/internal_000/paparazzi/
#ifndef DEPTH_SAVE_PATH
#define DEPTH_SAVE_PATH "/tmp/paparazzi/depth_latest.png"
#endif

// Model path — set per-target in airframe XML via DEPTH_MODEL_PATH define
#ifndef DEPTH_MODEL_PATH
#define DEPTH_MODEL_PATH "/tmp/paparazzi/model.onnx"
#endif
#define MODEL_PATH DEPTH_MODEL_PATH

// ABI message ID to publish depth result on.
// VISUAL_DETECTION is reused here for convenience:
//   pixel_x   = 0 (unused)
//   pixel_y   = 0 (unused)
//   pixel_width  = 0 (unused)
//   pixel_height = 0 (unused)
//   quality   = center_depth scaled to int (0-1000)
//   extra     = 0 (unused)
// If you define your own ABI message, change this accordingly.
#ifndef DEPTH_INFERENCE_ABI_ID
#define DEPTH_INFERENCE_ABI_ID 1
#endif
// ----------------------------------

// ------------- ONNX Runtime state (initialized once) -------------
static const OrtApi*        ort             = NULL;
static OrtEnv*              ort_env         = NULL;
static OrtSession*          ort_session     = NULL;
static OrtSessionOptions*   ort_options     = NULL;
static OrtMemoryInfo*       ort_memory_info = NULL;
static uint8_t              ort_ready       = 0;  // 1 once session is loaded

// ------------- Shared state between video thread and periodic -------------
static pthread_mutex_t      result_mutex;
static struct depth_result_t latest_result;

// ------------- Background inference thread -------------
static pthread_t            inference_thread;
static pthread_mutex_t      frame_mutex;
static pthread_cond_t       frame_cond;

// Double buffer: video callback writes to pending_frame,
// inference thread picks it up when ready
static float*               pending_frame   = NULL;  // float tensor [3 * H * W]
static uint8_t              frame_pending   = 0;     // 1 if a new frame is waiting
static uint8_t              module_running  = 1;     // set to 0 on shutdown

// ------------- Save timing -------------
static time_t last_save_time = 0;  // timestamp of last saved image

// ------------- Forward declarations -------------
static struct image_t* depth_inference_cb(struct image_t* img, uint8_t camera_id);
static void*           inference_thread_func(void* arg);
static void            run_inference(float* input_tensor_values);
static void            yuv422_to_float_tensor(struct image_t* img, float* tensor);
static void            save_depth_image(float* output_data, int out_pixels,
                                        float min_val, float range);

// =============================================================================
// INIT
// =============================================================================
void depth_inference_init(void)
{
    // Initialize mutexes and condition variable
    pthread_mutex_init(&result_mutex, NULL);
    pthread_mutex_init(&frame_mutex, NULL);
    pthread_cond_init(&frame_cond, NULL);

    memset(&latest_result, 0, sizeof(latest_result));

    // Allocate the pending frame buffer (float tensor: 3 channels * H * W)
    pending_frame = malloc(sizeof(float) * 3 * TARGET_HEIGHT * TARGET_WIDTH);
    if (!pending_frame) {
        printf("[depth_inference] ERROR: failed to allocate frame buffer\n");
        return;
    }

    // Load ONNX Runtime and model
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) {
        printf("[depth_inference] ERROR: failed to get ORT API\n");
        return;
    }

    OrtStatus* status = NULL;

    status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "depth_inference", &ort_env);
    if (status != NULL) {
        printf("[depth_inference] ERROR creating ORT env: %s\n", ort->GetErrorMessage(status));
        return;
    }

    status = ort->CreateSessionOptions(&ort_options);
    if (status != NULL) {
        printf("[depth_inference] ERROR creating session options\n");
        return;
    }

    // Single thread to avoid overloading the Bebop CPU
    ort->SetIntraOpNumThreads(ort_options, 1);
    ort->SetInterOpNumThreads(ort_options, 1);

    status = ort->CreateSession(ort_env, MODEL_PATH, ort_options, &ort_session);
    if (status != NULL) {
        printf("[depth_inference] ERROR loading model: %s\n", ort->GetErrorMessage(status));
        return;
    }

    status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_memory_info);
    if (status != NULL) {
        printf("[depth_inference] ERROR creating memory info\n");
        return;
    }

    ort_ready = 1;
    printf("[depth_inference] Model loaded successfully from %s\n", MODEL_PATH);

    // Register video callback on front camera
    cv_add_to_device(&front_camera, depth_inference_cb, 4, 0);

    // Start background inference thread
    pthread_create(&inference_thread, NULL, inference_thread_func, NULL);
}

// =============================================================================
// PERIODIC (runs at 4Hz in autopilot thread)
// Reads latest result and publishes via ABI
// =============================================================================
void depth_inference_periodic(void)
{
    if (!ort_ready) return;

    struct depth_result_t local_result;

    pthread_mutex_lock(&result_mutex);
    if (!latest_result.fresh) {
        pthread_mutex_unlock(&result_mutex);
        return;  // No new result since last read
    }
    local_result = latest_result;
    latest_result.fresh = 0;
    pthread_mutex_unlock(&result_mutex);

    // Publish via ABI — reusing VISUAL_DETECTION message
    // quality field carries center_depth scaled 0-1000
    int32_t depth_scaled = (int32_t)(local_result.center_depth * 1000.0f);
    AbiSendMsgVISUAL_DETECTION(
        DEPTH_INFERENCE_ABI_ID,
        0,            // pixel_x   (unused)
        0,            // pixel_y   (unused)
        0,            // pixel_width (unused)
        0,            // pixel_height (unused)
        depth_scaled, // quality: center depth 0-1000
        0             // extra (unused)
    );

    printf("[depth_inference] Published depth: center=%.3f min=%.3f max=%.3f\n",
           local_result.center_depth,
           local_result.min_depth,
           local_result.max_depth);
}

// =============================================================================
// VIDEO CALLBACK (runs in video thread)
// Converts frame to float tensor and signals inference thread
// =============================================================================
static struct image_t* depth_inference_cb(struct image_t* img, uint8_t camera_id)
{
    if (!ort_ready || !pending_frame) return img;

    // Only queue a new frame if the inference thread is not busy
    // (trylock: if we can't get the lock immediately, skip this frame)
    if (pthread_mutex_trylock(&frame_mutex) != 0) {
        return img;  // Inference thread is busy, drop this frame
    }

    if (!frame_pending) {
        // Convert YUV422 camera image to float RGB tensor
        yuv422_to_float_tensor(img, pending_frame);
        frame_pending = 1;
        pthread_cond_signal(&frame_cond);
    }

    pthread_mutex_unlock(&frame_mutex);
    return img;
}

// =============================================================================
// INFERENCE THREAD
// Waits for new frames and runs ONNX inference
// =============================================================================
static void* inference_thread_func(void* arg)
{
    (void)arg;

    // Allocate a local copy of the input tensor
    float* local_tensor = malloc(sizeof(float) * 3 * TARGET_HEIGHT * TARGET_WIDTH);
    if (!local_tensor) {
        printf("[depth_inference] ERROR: inference thread failed to allocate buffer\n");
        return NULL;
    }

    while (module_running) {
        // Wait for a new frame
        pthread_mutex_lock(&frame_mutex);
        while (!frame_pending && module_running) {
            pthread_cond_wait(&frame_cond, &frame_mutex);
        }

        if (!module_running) {
            pthread_mutex_unlock(&frame_mutex);
            break;
        }

        // Copy frame and clear pending flag quickly so video thread can queue next frame
        memcpy(local_tensor, pending_frame, sizeof(float) * 3 * TARGET_HEIGHT * TARGET_WIDTH);
        frame_pending = 0;
        pthread_mutex_unlock(&frame_mutex);

        // Run inference (this can take several seconds on the Bebop)
        run_inference(local_tensor);
    }

    free(local_tensor);
    return NULL;
}

// =============================================================================
// RUN INFERENCE
// Takes float tensor [3 * H * W], runs ONNX model, stores result
// =============================================================================
static void run_inference(float* input_tensor_values)
{
    int out_pixels = TARGET_HEIGHT * TARGET_WIDTH;
    int64_t input_shape[4] = {1, 3, TARGET_HEIGHT, TARGET_WIDTH};

    OrtValue* input_tensor  = NULL;
    OrtValue* output_tensor = NULL;
    OrtStatus* status       = NULL;

    status = ort->CreateTensorWithDataAsOrtValue(
        ort_memory_info,
        input_tensor_values,
        sizeof(float) * 3 * out_pixels,
        input_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    if (status != NULL) {
        printf("[depth_inference] ERROR creating input tensor: %s\n", ort->GetErrorMessage(status));
        return;
    }

    const char* input_names[]  = {"input"};
    const char* output_names[] = {"output"};

    status = ort->Run(
        ort_session,
        NULL,
        input_names,
        (const OrtValue* const*)&input_tensor,
        1,
        output_names,
        1,
        &output_tensor
    );
    if (status != NULL) {
        printf("[depth_inference] ERROR during inference: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseValue(input_tensor);
        return;
    }

    // Get output data
    float* output_data = NULL;
    ort->GetTensorMutableData(output_tensor, (void**)&output_data);

    // Compute stats: min, max, center region average
    float min_val = output_data[0];
    float max_val = output_data[0];
    for (int i = 0; i < out_pixels; i++) {
        if (output_data[i] < min_val) min_val = output_data[i];
        if (output_data[i] > max_val) max_val = output_data[i];
    }

    float range = max_val - min_val;
    if (range < 1e-6f) range = 1.0f;

    // Average depth in center 1/3 of the image (most relevant for obstacle avoidance)
    int center_x_start = TARGET_WIDTH  / 3;
    int center_x_end   = TARGET_WIDTH  * 2 / 3;
    int center_y_start = TARGET_HEIGHT / 3;
    int center_y_end   = TARGET_HEIGHT * 2 / 3;

    float center_sum   = 0.0f;
    int   center_count = 0;

    for (int y = center_y_start; y < center_y_end; y++) {
        for (int x = center_x_start; x < center_x_end; x++) {
            float normalized = (output_data[y * TARGET_WIDTH + x] - min_val) / range;
            center_sum += normalized;
            center_count++;
        }
    }

    float center_depth = (center_count > 0) ? (center_sum / center_count) : 0.0f;

    // Save depth image every DEPTH_SAVE_INTERVAL_SEC seconds
    save_depth_image(output_data, out_pixels, min_val, range);

    // Store result (protected by mutex)
    pthread_mutex_lock(&result_mutex);
    latest_result.center_depth = center_depth;
    latest_result.min_depth    = min_val;
    latest_result.max_depth    = max_val;
    latest_result.fresh        = 1;
    pthread_mutex_unlock(&result_mutex);

    ort->ReleaseValue(output_tensor);
    ort->ReleaseValue(input_tensor);
}

// =============================================================================
// SAVE DEPTH IMAGE
// Converts float depth output to grayscale PNG and saves to DEPTH_SAVE_PATH.
// Only runs every DEPTH_SAVE_INTERVAL_SEC seconds so it doesn't spam disk.
// In simulation: look for the file at /tmp/paparazzi/depth_latest.png
// =============================================================================
static void save_depth_image(float* output_data, int out_pixels,
                              float min_val, float range)
{
    time_t now = time(NULL);
    if ((now - last_save_time) < DEPTH_SAVE_INTERVAL_SEC) {
        return;  // Not enough time has passed yet
    }
    last_save_time = now;

    // Convert normalized float depth to 8-bit grayscale
    unsigned char* depth_img = malloc(out_pixels);
    if (!depth_img) {
        printf("[depth_inference] ERROR: failed to allocate depth save buffer\n");
        return;
    }

    for (int i = 0; i < out_pixels; i++) {
        float normalized = (output_data[i] - min_val) / range;
        normalized = normalized < 0.0f ? 0.0f : (normalized > 1.0f ? 1.0f : normalized);
        depth_img[i] = (unsigned char)(normalized * 255.0f);
    }

    // Write grayscale PNG (1 channel)
    int result = stbi_write_png(DEPTH_SAVE_PATH,
                                TARGET_WIDTH, TARGET_HEIGHT,
                                1,           // 1 channel = grayscale
                                depth_img,
                                TARGET_WIDTH);
    if (result) {
        printf("[depth_inference] Saved depth image to %s\n", DEPTH_SAVE_PATH);
    } else {
        printf("[depth_inference] ERROR: failed to save depth image to %s\n", DEPTH_SAVE_PATH);
    }

    free(depth_img);
}

// =============================================================================
// YUV422 -> FLOAT RGB TENSOR
// Converts Paparazzi UYVY image to normalized float [3 * H * W] tensor
// Also resizes from camera resolution to TARGET_WIDTH x TARGET_HEIGHT
// (simple nearest-neighbour for speed)
// =============================================================================
static void yuv422_to_float_tensor(struct image_t* img, float* tensor)
{
    int src_w = img->w;
    int src_h = img->h;
    uint8_t* buf = (uint8_t*)img->buf;

    int out_pixels = TARGET_HEIGHT * TARGET_WIDTH;

    for (int ty = 0; ty < TARGET_HEIGHT; ty++) {
        for (int tx = 0; tx < TARGET_WIDTH; tx++) {

            // Nearest-neighbour scale
            int sx = tx * src_w / TARGET_WIDTH;
            int sy = ty * src_h / TARGET_HEIGHT;

            // UYVY packing: for pixel at (sx, sy)
            // Each pair of pixels = 4 bytes: U Y0 V Y1
            int pair_idx  = sy * src_w + (sx & ~1);  // round down to even pixel
            int byte_idx  = pair_idx * 2;             // 2 bytes per pixel in UYVY

            uint8_t u = buf[byte_idx + 0];
            uint8_t y = buf[byte_idx + 1 + (sx & 1) * 2];  // Y0 or Y1
            uint8_t v = buf[byte_idx + 2];

            // YUV -> RGB conversion
            float yf = (float)y;
            float uf = (float)u - 128.0f;
            float vf = (float)v - 128.0f;

            float r = yf + 1.402f   * vf;
            float g = yf - 0.344f   * uf - 0.714f * vf;
            float b = yf + 1.772f   * uf;

            // Clamp and normalize to [0, 1]
            r = r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r);
            g = g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g);
            b = b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b);

            int out_idx = ty * TARGET_WIDTH + tx;
            tensor[0 * out_pixels + out_idx] = r / 255.0f;  // R channel
            tensor[1 * out_pixels + out_idx] = g / 255.0f;  // G channel
            tensor[2 * out_pixels + out_idx] = b / 255.0f;  // B channel
        }
    }
}
