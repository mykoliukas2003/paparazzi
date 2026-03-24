#ifndef CV_OUTPUT_H_
#define CV_OUTPUT_H_

#include <stdbool.h>
#include <stdint.h>

#define CV_OUTPUT_NUM_BLOCKS 7

extern float cv_output_nav_vector[CV_OUTPUT_NUM_BLOCKS];
extern bool cv_output_nav_valid;

extern void cv_output_init(void);
extern void cv_output_close(void);

#endif /* CV_OUTPUT_H_ */
