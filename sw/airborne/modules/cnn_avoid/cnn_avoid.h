#ifndef CNN_AVOID_H_
#define CNN_AVOID_H_
 
#include <stdint.h>
 
extern void cnn_avoid_init(void);
extern void cnn_avoid_periodic(void);
extern void cnn_avoid_set_trajectory(uint8_t traj_id);
 
/* Avoidance tunables */
extern float cnn_safe_distance_threshold;
extern float cnn_caution_distance_threshold;
extern float cnn_caution_exit_threshold;
extern float cnn_depth_filter_alpha;
extern float cnn_max_distance;
 
/* Trajectory tunables */
extern float traj_circle_radius;
extern float traj_circle_speed;
extern float traj_eight_scale;
extern float traj_lawn_step;
 
#endif /* CNN_AVOID_H_ */