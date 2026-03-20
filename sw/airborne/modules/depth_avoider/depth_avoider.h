/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/depth_avoider/depth_avoider.h"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 */

#ifndef DEPTH_AVOIDER_H
#define DEPTH_AVOIDER_H

// settings
extern float oa_color_count_frac;
extern float safe_distance_threshold;

// functions
extern void depth_avoider_init(void);
extern void depth_avoider_periodic(void);

#endif

