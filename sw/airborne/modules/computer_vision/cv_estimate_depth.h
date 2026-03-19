/*
 * Copyright (C) 2019 Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of Paparazzi.
 *
 * Paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * Paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * @file modules/computer_vision/cv_estimate_depth.h
 * Estimates the depth of objects in the image
 */

#ifndef DEPTH_ESTIMATOR_CV_H
#define DEPTH_ESTIMATOR_CV_H

#include <stdint.h>
#include <stdbool.h>

// Module settings
/*
extern uint8_t cod_lum_min1;
extern uint8_t cod_lum_max1;
extern uint8_t cod_cb_min1;
extern uint8_t cod_cb_max1;
extern uint8_t cod_cr_min1;
extern uint8_t cod_cr_max1;

extern uint8_t cod_lum_min2;
extern uint8_t cod_lum_max2;
extern uint8_t cod_cb_min2;
extern uint8_t cod_cb_max2;
extern uint8_t cod_cr_min2;
extern uint8_t cod_cr_max2;
*/
extern bool depth_draw;
extern uint8_t depth_threshold;

// Module functions
extern void estimate_depth_init(void);
extern void estimate_depth_periodic(void);

#endif /* DEPTH_ESTIMATOR_CV_H */
