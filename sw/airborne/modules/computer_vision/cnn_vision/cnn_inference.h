#ifndef CNN_INFERENCE_H_
#define CNN_INFERENCE_H_

/**
 * Pure-C forward pass for the depth estimation CNN.
 * Architecture: 3 conv (encoder) + 3 deconv (decoder) with skip connections.
 * Channels: 3→8→16→32 → 16→8→1
 * No external dependencies (no ONNX Runtime).
 *
 * Input:  float[3][80][520]  — normalised YUV image (NCHW)
 * Output: float[1][80][520]  — depth map
 */

#define CNN_IN_C   3
#define CNN_H      80
#define CNN_W      520
#define CNN_HALF_H 40
#define CNN_HALF_W 260

/**
 * @brief Run the depth CNN forward pass.
 * @param input   Normalised input image, NCHW layout [3][80][520], values in [0,1]
 * @param output  Depth map output [80][520]
 */
void cnn_forward(const float input[CNN_IN_C][CNN_H][CNN_W],
                 float output[CNN_H][CNN_W]);

#endif /* CNN_INFERENCE_H_ */
