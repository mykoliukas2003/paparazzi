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

// Own header
#include "modules/computer_vision/cv_estimate_depth.h"
#include "modules/computer_vision/depth_estim/depth_estimator.h"
//add the onnx2c output file possibly
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

#define DEPTH_ESTIMATOR_VERBOSE FALSE
#define PRINT(string,...) fprintf(stderr, "[depth_estimator->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if DEPTH_ESTIMATOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif
/*
#ifndef DEPTH_CAMERA
#define DEPTH_CAMERA "front_camera" ///< Default camera to run depth estimation on
#endif
*/
#ifndef DEPTH_CAMERA_FPS
#define DEPTH_CAMERA_FPS 0 ///< Default FPS (zero means run at camera fps)
#endif

static pthread_mutex_t mutex;
// Filter Settings
/*
uint8_t cod_lum_min1 = 0;
uint8_t cod_lum_max1 = 0;
uint8_t cod_cb_min1 = 0;
uint8_t cod_cb_max1 = 0;
uint8_t cod_cr_min1 = 0;
uint8_t cod_cr_max1 = 0;

uint8_t cod_lum_min2 = 0;
uint8_t cod_lum_max2 = 0;
uint8_t cod_cb_min2 = 0;
uint8_t cod_cb_max2 = 0;
uint8_t cod_cr_min2 = 0;
uint8_t cod_cr_max2 = 0;
*/
bool depth_draw = true;
uint8_t depth_threshold = 128;

// define global variables
struct color_object_t {
  int32_t x_c;
  int32_t y_c;
  uint32_t color_count;
  bool updated;
};
struct color_object_t global_filters[2];
struct depths_t {
  float depth1;
  float depth2;
  float depth3;
  bool updated;
};
struct depths_t global_depths;

// Function
uint32_t find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
                              uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max);

/*
 * object_detector
 * @param img - input image to process
 * @param filter - which detection filter to process
 * @return img
 */
/*
static struct image_t *object_detector(struct image_t *img, uint8_t filter)
{
  uint8_t lum_min, lum_max;
  uint8_t cb_min, cb_max;
  uint8_t cr_min, cr_max;
  bool draw;

  switch (filter){
    case 1:
      lum_min = cod_lum_min1;
      lum_max = cod_lum_max1;
      cb_min = cod_cb_min1;
      cb_max = cod_cb_max1;
      cr_min = cod_cr_min1;
      cr_max = cod_cr_max1;
      draw = cod_draw1;
      break;
    case 2:
      lum_min = cod_lum_min2;
      lum_max = cod_lum_max2;
      cb_min = cod_cb_min2;
      cb_max = cod_cb_max2;
      cr_min = cod_cr_min2;
      cr_max = cod_cr_max2;
      draw = cod_draw2;
      break;
    default:
      return img;
  };

  int32_t x_c, y_c;

  // Filter and find centroid
  uint32_t count = find_object_centroid(img, &x_c, &y_c, draw, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max);
  VERBOSE_PRINT("Color count %d: %u, threshold %u, x_c %d, y_c %d\n", camera, object_count, count_threshold, x_c, y_c);
  VERBOSE_PRINT("centroid %d: (%d, %d) r: %4.2f a: %4.2f\n", camera, x_c, y_c,
        hypotf(x_c, y_c) / hypotf(img->w * 0.5, img->h * 0.5), RadOfDeg(atan2f(y_c, x_c)));

  pthread_mutex_lock(&mutex);
  global_filters[filter-1].color_count = count;
  global_filters[filter-1].x_c = x_c;
  global_filters[filter-1].y_c = y_c;
  global_filters[filter-1].updated = true;
  pthread_mutex_unlock(&mutex);

  return img;
}
*/
static struct image_t *depth_estim(struct image_t *img)
{
  /*
  uint8_t lum_min, lum_max;
  uint8_t cb_min, cb_max;
  uint8_t cr_min, cr_max;
  bool draw;


  int32_t x_c, y_c;
  */
  // call on the model
  //printf("Running depth estimation on image of size %d x %d\n", img->w, img->h);
  
  
  //if needed reshape the image to fit the model input dimensions.
  //also convert to float
  //int img_w = img->w;
  //int img_h = img->h;
  uint8_t *img_buf = (uint8_t *)img->buf;
  float tensor_input[1][249600];
  for(int i = 0; i < 249600; i++){
    tensor_input[0][i] = (float)img_buf[i];
  }
  


  float tensor_output[1][3];
  //printf(tensor_input);

  entry(tensor_input, tensor_output);
  //printf("model output: %f, %f, %f\n", tensor_output[0][0], tensor_output[0][1], tensor_output[0][2]);
  VERBOSE_PRINT("Model output: %f, %f, %f\n", tensor_output[0][0], tensor_output[0][1], tensor_output[0][2]);
  
  pthread_mutex_lock(&mutex);
  global_depths.depth1 = tensor_output[0][0];
  global_depths.depth2 = tensor_output[0][1];
  global_depths.depth3 = tensor_output[0][2];
  global_depths.updated = true;
  pthread_mutex_unlock(&mutex);
  
  /*
  pthread_mutex_lock(&mutex);
  global_filters[filter-1].color_count = count;
  global_filters[filter-1].x_c = x_c;
  global_filters[filter-1].y_c = y_c;
  global_filters[filter-1].updated = true;
  pthread_mutex_unlock(&mutex);
  */
  // make the square where the model estimates the depth brighter in the image for debugging
  if(depth_draw){
    for(int x = 50; x < 101; x++){
      //bbox left
      for(int y = 45; y < 45+150+1; y++){
        int idx = y * img->w + x;
        img_buf[idx*2+1] = (int)tensor_output[0][0];
      }
      //bbox center
      for(int y = 45+150+1; y < 45+150+1+150+1; y++){
        int idx = y * img->w + x;
        img_buf[idx*2+1] = (int)tensor_output[0][1];
      }
      //bbox right
      for(int y = 45+150+1+150+1; y < 45+150+1+150+1+150+1; y++){
        int idx = y * img->w + x;
        img_buf[idx*2+1] = (int)tensor_output[0][2];
      }
    }
  }
  return img;
}
//try if the wrapper is the issue
struct image_t *depth_estim1(struct image_t *img, uint8_t camera_id);
struct image_t *depth_estim1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return depth_estim(img);
}
/*
struct image_t *object_detector1(struct image_t *img, uint8_t camera_id);
struct image_t *object_detector1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return object_detector(img, 1);
}

struct image_t *object_detector2(struct image_t *img, uint8_t camera_id);
struct image_t *object_detector2(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return object_detector(img, 2);
}
*/
void estimate_depth_init(void)
{
  /*
  memset(global_filters, 0, 2*sizeof(struct color_object_t));
  pthread_mutex_init(&mutex, NULL);
*/
  
  memset(&global_depths, 0, sizeof(struct depths_t));
  pthread_mutex_init(&mutex, NULL);
  
 printf("Initializing depth estimation module\n");
 //printf("Depth camera: %s, Depth camera FPS: %d, Depth draw: %d, Depth threshold: %d\n", DEPTH_CAMERA, DEPTH_CAMERA_FPS, depth_draw, depth_threshold);
#ifdef DEPTH_CAMERA
#ifdef DEPTH_DRAW
  depth_draw = DEPTH_DRAW;
#endif

  cv_add_to_device(&DEPTH_CAMERA, depth_estim1, DEPTH_CAMERA_FPS, 0);
#endif
#ifdef DEPTH_THRESHOLD
  depth_threshold = DEPTH_THRESHOLD;
#endif
/*
#ifdef COLOR_OBJECT_DETECTOR_CAMERA2
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN2
  cod_lum_min2 = COLOR_OBJECT_DETECTOR_LUM_MIN2;
  cod_lum_max2 = COLOR_OBJECT_DETECTOR_LUM_MAX2;
  cod_cb_min2 = COLOR_OBJECT_DETECTOR_CB_MIN2;
  cod_cb_max2 = COLOR_OBJECT_DETECTOR_CB_MAX2;
  cod_cr_min2 = COLOR_OBJECT_DETECTOR_CR_MIN2;
  cod_cr_max2 = COLOR_OBJECT_DETECTOR_CR_MAX2;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW2
  cod_draw2 = COLOR_OBJECT_DETECTOR_DRAW2;
#endif

  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA2, object_detector2, COLOR_OBJECT_DETECTOR_FPS2, 1);
#endif
*/
}



void estimate_depth_periodic(void)
{
  static struct depths_t local_depths;
  pthread_mutex_lock(&mutex);
  memcpy(&local_depths, &global_depths, sizeof(struct depths_t));
  global_depths.updated = false; // set to false since we have now copied the updated value to local variable
  pthread_mutex_unlock(&mutex);

  if(local_depths.updated){
    int16_t send_left = (int16_t)(local_depths.depth1 * 100.0f);
    int16_t send_straight = (int16_t)(local_depths.depth2 * 100.0f);
    int16_t send_right = (int16_t)(local_depths.depth3 * 100.0f);
    VERBOSE_PRINT("[Depth Estimator] Sending ABI -> L: %d, S: %d, R: %d\n", send_left, send_straight, send_right);

    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION1_ID, send_left, send_straight, send_right, 0, 0, 0);
    //does this save to memory? shouldnt I use a mutex to set this to false? now done like in an example in the cv_detect_color_object.c file
    //local_depths.updated = false; // we dont update this since global takes care of this
  }
}
