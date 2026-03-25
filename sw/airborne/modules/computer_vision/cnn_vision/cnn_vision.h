#ifndef CNN_VISION_H_
#define CNN_VISION_H_
 
#include <stdbool.h>
#include <stdint.h>
 
#define CNN_VISION_NUM_BLOCKS 7
 
/** Per-block average depth. Updated by the vision thread. */
extern float cnn_vision_nav_vector[CNN_VISION_NUM_BLOCKS];
 
/** Set to true once the first valid inference has completed. */
extern bool cnn_vision_nav_valid;
 
extern void cnn_vision_init(void);
extern void cnn_vision_close(void);
 
#endif /* CNN_VISION_H_ */