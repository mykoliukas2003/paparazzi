/*
 * Orange Nav Avoider Module
 * Combines orange pixel-based obstacle detection with trajectory following.
 */

#ifndef ORANGE_NAV_AVOIDER_H
#define ORANGE_NAV_AVOIDER_H

#include <stdint.h>
#include <stdbool.h>

/* Tunable settings */
extern float oa_color_count_frac;
extern float maxDistance;
extern float traj_circle_radius;
extern float traj_eight_scale;
extern float maxheading_increment;

/* State visibility */
extern int32_t color_count;
extern int16_t obstacle_free_confidence;

/* Module functions */
extern void orange_nav_avoider_init(void);
extern void orange_nav_avoider_periodic(void);
extern void orange_nav_avoider_trajectory(uint8_t traj_id);

#endif /* ORANGE_NAV_AVOIDER_H */