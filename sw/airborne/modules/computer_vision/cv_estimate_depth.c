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

#define DEPTH_ESTIMATOR_VERBOSE TRUE
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
bool depth_draw = false;
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
  printf("Running depth estimation on image of size %d x %d\n", img->w, img->h);
  
  
  //if needed reshape the image to fit the model input dimensions.
  //also convert to float
  int img_w = img->w;
  int img_h = img->h;
  uint8_t *img_buf = (uint8_t *)img->buf;
  float input[1][3][520][240];
  for(int i = 0; i < img->w * img->h * 2; i+=2){
    uint8_t u = img_buf[i];
    uint8_t y = img_buf[i+1];
    uint8_t v = img_buf[i+2];//just wrong but for testing the wrapper
    int pixel_index = i/2;
    int x = pixel_index % img_w;
    int y_coord = pixel_index / img_w;
    input[0][0][x][y_coord] = (float) y;
    input[0][1][x][y_coord] = (float) u;
    input[0][2][x][y_coord] = (float) v;
  }


  float tensor_output[1][3];
  entry(input, tensor_output);
  printf("model output: %f, %f, %f\n", tensor_output[0][0], tensor_output[0][1], tensor_output[0][2]);
  
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
  
  memset(global_filters, 0, 2*sizeof(struct color_object_t));
  pthread_mutex_init(&mutex, NULL);

  /*
  memset(global_depths, 0, sizeof(struct depths_t));
  pthread_mutex_init(&mutex, NULL);
  */
 printf("Initializing depth estimation module\n");
 printf("Depth camera: %s, Depth camera FPS: %d, Depth draw: %d, Depth threshold: %d\n", DEPTH_CAMERA, DEPTH_CAMERA_FPS, depth_draw, depth_threshold);
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

/*
 * find_object_centroid
 *
 * Finds the centroid of pixels in an image within filter bounds.
 * Also returns the amount of pixels that satisfy these filter bounds.
 *
 * @param img - input image to process formatted as YUV422.
 * @param p_xc - x coordinate of the centroid of color object
 * @param p_yc - y coordinate of the centroid of color object
 * @param lum_min - minimum y value for the filter in YCbCr colorspace
 * @param lum_max - maximum y value for the filter in YCbCr colorspace
 * @param cb_min - minimum cb value for the filter in YCbCr colorspace
 * @param cb_max - maximum cb value for the filter in YCbCr colorspace
 * @param cr_min - minimum cr value for the filter in YCbCr colorspace
 * @param cr_max - maximum cr value for the filter in YCbCr colorspace
 * @param draw - whether or not to draw on image
 * @return number of pixels of image within the filter bounds.
 */
uint32_t find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
                              uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max)
{
  uint32_t cnt = 0;
  uint32_t tot_x = 0;
  uint32_t tot_y = 0;
  uint8_t *buffer = img->buf;

  // Go through all the pixels
  for (uint16_t y = 0; y < img->h; y++) {
    for (uint16_t x = 0; x < img->w; x ++) {
      // Check if the color is inside the specified values
      uint8_t *yp, *up, *vp;
      if (x % 2 == 0) {
        // Even x
        up = &buffer[y * 2 * img->w + 2 * x];      // U
        yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y1
        vp = &buffer[y * 2 * img->w + 2 * x + 2];  // V
        //yp = &buffer[y * 2 * img->w + 2 * x + 3]; // Y2
      } else {
        // Uneven x
        up = &buffer[y * 2 * img->w + 2 * x - 2];  // U
        //yp = &buffer[y * 2 * img->w + 2 * x - 1]; // Y1
        vp = &buffer[y * 2 * img->w + 2 * x];      // V
        yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y2
      }
      if ( (*yp >= lum_min) && (*yp <= lum_max) &&
           (*up >= cb_min ) && (*up <= cb_max ) &&
           (*vp >= cr_min ) && (*vp <= cr_max )) {
        cnt ++;
        tot_x += x;
        tot_y += y;
        if (draw){
          *yp = 255;  // make pixel brighter in image
        }
      }
    }
  }
  if (cnt > 0) {
    *p_xc = (int32_t)roundf(tot_x / ((float) cnt) - img->w * 0.5f);
    *p_yc = (int32_t)roundf(img->h * 0.5f - tot_y / ((float) cnt));
  } else {
    *p_xc = 0;
    *p_yc = 0;
  }
  return cnt;
}

void estimate_depth_periodic(void)
{
  static struct depths_t local_depths;
  pthread_mutex_lock(&mutex);
  memcpy(&local_depths, &global_depths, sizeof(struct depths_t));
  global_depths.updated = false; // set to false since we have now copied the updated value to local variable
  pthread_mutex_unlock(&mutex);

  if(local_depths.updated){
    int16_t send_left = (int16_t)(local_depths.depth1 * 1000.0f);
    int16_t send_straight = (int16_t)(local_depths.depth2 * 1000.0f);
    int16_t send_right = (int16_t)(local_depths.depth3 * 1000.0f);
    printf("[Depth Estimator] Sending ABI -> L: %d, S: %d, R: %d\n", send_left, send_straight, send_right);

    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION1_ID, send_left, send_straight, send_right, 0, 0, 0);
    //does this save to memory? shouldnt I use a mutex to set this to false? now done like in an example in the cv_detect_color_object.c file
    //local_depths.updated = false; // we dont update this since global takes care of this
  }
}
