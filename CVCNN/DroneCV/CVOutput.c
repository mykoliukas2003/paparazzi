// load bebop data & crop

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>  // ONNX Runtime C API
#include <stdint.h>
#include <float.h>

#define STRIP_HEIGHT 80
#define STRIP_WIDTH 520      // Example: horizontal crop width
#define NUM_BLOCKS 7         // Number of vertical blocks to segment output
#define CHANNELS 3           // RGB

uint8_t*** capture_bebop_image() {
    // TODO: implement Bebop SDK camera capture
    return NULL;
}

// Crop central horizontal strip from image
void crop_strip(uint8_t*** image, uint8_t strip[STRIP_HEIGHT][STRIP_WIDTH][CHANNELS], int img_height, int img_width) {
    int start_x = (img_width - STRIP_WIDTH) / 2;
    int start_y = (img_height - STRIP_HEIGHT) / 2;

    for(int y = 0; y < STRIP_HEIGHT; y++){
        for(int x = 0; x < STRIP_WIDTH; x++){
            for(int c = 0; c < CHANNELS; c++){
                strip[y][x][c] = image[start_y + y][start_x + x][c];
            }
        }
    }
}

void normalize_strip(uint8_t strip[STRIP_HEIGHT][STRIP_WIDTH][CHANNELS], float input[1][CHANNELS][STRIP_HEIGHT][STRIP_WIDTH]) {
    for(int c=0; c<CHANNELS; c++){
        for(int y=0; y<STRIP_HEIGHT; y++){
            for(int x=0; x<STRIP_WIDTH; x++){
                input[0][c][y][x] = strip[y][x][c] / 255.0f;
            }
        }
    }
}

// load img into CNN (onnx)

void run_cnn(OrtSession* session, float input[1][CHANNELS][STRIP_HEIGHT][STRIP_WIDTH], float output[STRIP_HEIGHT][STRIP_WIDTH]){
    // TODO: use ONNX Runtime C API
    // Create input tensor from `input`
    // Run OrtRun()
    // Fill `output` with depth map values
}


// Now have depth map output from CNN. Segment into 7 blocks & give output
// Map the furthest block to navigation labels
const char* nav_labels[NUM_BLOCKS] = {"LL", "FL", "L", "C", "R", "FR", "RR"};

int pick_furthest_block(float depth[STRIP_HEIGHT][STRIP_WIDTH]) {
    int block_width = STRIP_WIDTH / NUM_BLOCKS;
    float max_val = -FLT_MAX;
    int furthest_block = 0;

    for(int b = 0; b < NUM_BLOCKS; b++){
        int start_x = b * block_width;
        int end_x = start_x + block_width;

        for(int y = 0; y < STRIP_HEIGHT; y++){
            for(int x = start_x; x < end_x; x++){
                if(depth[y][x] > max_val){   // largest = furthest
                    max_val = depth[y][x];
                    furthest_block = b;
                }
            }
        }
    }

    return furthest_block; // index 0..6
}

int main() {
    // Capture image
    uint8_t*** image = capture_bebop_image();
    if(!image){
        printf("Error: failed to capture image!\n");
        return -1;
    }

    // Crop strip
    uint8_t strip[STRIP_HEIGHT][STRIP_WIDTH][CHANNELS];
    crop_strip(image, strip, 240, 520); // example full image size 320x240

    // Normalize
    float cnn_input[1][CHANNELS][STRIP_HEIGHT][STRIP_WIDTH];
    normalize_strip(strip, cnn_input);

    // Load ONNX model and run CNN
    OrtEnv* env;
    OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "bebop_depth", &env);
    OrtSessionOptions* session_options;
    OrtCreateSessionOptions(&session_options);
    OrtSession* session;
    OrtCreateSession(env, "depth_model.onnx", session_options, &session);

    float depth_map[STRIP_HEIGHT][STRIP_WIDTH];
    run_cnn(session, cnn_input, depth_map);

    int furthest_block = pick_furthest_block(depth_map);
    printf("Furthest direction: %s\n", nav_labels[furthest_block]);

    // Cleanup ONNX Runtime
    OrtReleaseSession(session);
    OrtReleaseSessionOptions(session_options);
    OrtReleaseEnv(env);

    return 0;
}