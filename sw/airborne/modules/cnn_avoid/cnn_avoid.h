#ifndef CNN_AVOID_H_
#define CNN_AVOID_H_
 
#include <stdint.h>
 
extern void cnn_avoid_init(void);
extern void cnn_avoid_periodic(void);
extern void cnn_avoid_set_trajectory(uint8_t traj_id);
 
/* Tunables — can be overridden from airframe XML via <define> */
extern float cnn_safe_distance_threshold;
extern float cnn_caution_distance_threshold;
extern float cnn_caution_exit_threshold;
extern float cnn_max_distance;
 
#endif /* CNN_AVOID_H_ */