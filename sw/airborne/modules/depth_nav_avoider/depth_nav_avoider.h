/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/depth_nav_avoider/depth_nav_avoider.h"
 * @author Roland Meertens (original), extended for combined depth + confidence + trajectory avoidance
 *
 * Combined avoider module for the CyberZoo obstacle avoidance challenge.
 * Uses depth estimates (left/straight/right) from cv_estimate_depth combined
 * with a confidence-based state machine and trajectory following.
 */

#ifndef DEPTH_NAV_AVOIDER_H
#define DEPTH_NAV_AVOIDER_H

#include <stdint.h>
#include <stdbool.h>

/* ================================ */
/*    Tunable Settings              */
/* ================================ */

/** Depth below which a full stop and rotation is triggered (hard obstacle) */
extern float safe_distance_threshold;

/** Depth below which the drone slows down and begins cautious turning */
extern float caution_distance_threshold;

/** Maximum waypoint displacement per cycle [m] */
extern float maxDistance;

/** Circle trajectory radius [m] */
extern float traj_circle_radius;
extern float traj_circle_speed;
extern float depth_filter_alpha;
/** Figure-eight / lawnmower scale [m] */

extern float TURN_BASE_DEG;
extern float RANDOM_EXTRA_DEG;

/* ================================ */
/*    State Visibility (read-only)  */
/* ================================ */

/** Current raw depth readings from cv_estimate_depth */
extern float depth_left;
extern float depth_straight;
extern float depth_right;

/** Confidence that path ahead is obstacle-free (0 to max_trajectory_confidence) */
extern int16_t obstacle_free_confidence;

/* ================================ */
/*    Module Functions              */
/* ================================ */

/** Initialises the avoider: seeds RNG, reads WP_CENTER, binds ABI callback */
extern void depth_nav_avoider_init(void);

/** Main periodic function: filters depth, updates confidence, runs state machine */
extern void depth_nav_avoider_periodic(void);

/**
 * @brief Selects which trajectory to follow.
 * @param traj_id  0 = circle, 1 = figure-eight, 2 = lawnmower
 * Called from flight plan blocks to switch trajectory at runtime.
 */
extern void depth_nav_set_trajectory(uint8_t traj_id);

#endif /* DEPTH_NAV_AVOIDER_H */