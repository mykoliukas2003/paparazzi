#ifndef DEPTH_INFERENCE_H
#define DEPTH_INFERENCE_H

#include <stdint.h>

/** Result struct — written by inference thread, read by periodic */
struct depth_result_t {
    float   center_depth;   /**< Mean normalised depth in centre 1/3 of image [0-1] */
    float   min_depth;      /**< Raw minimum value from model output */
    float   max_depth;      /**< Raw maximum value from model output */
    uint8_t fresh;          /**< 1 if this result has not been consumed yet */
};

/** Called once at autopilot startup */
extern void depth_inference_init(void);

/** Called periodically (4 Hz) from the autopilot thread.
 *  Reads the latest inference result and publishes it via ABI. */
extern void depth_inference_periodic(void);

#endif /* DEPTH_INFERENCE_H */
