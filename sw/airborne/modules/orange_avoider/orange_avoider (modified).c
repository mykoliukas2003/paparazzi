/**
 * @file "modules/orange_avoider/orange_avoider.c"
 * @author Roland Meertens
 */

#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>

#define NAV_C
#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string, ...) fprintf(stderr, "[orange_avoider->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// Define navigation states
enum navigation_state_t {
  SAFE,
  CAUTION,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
  };

// Define and initialise variables
enum navigation_state_t navigation_state = SAFE; // Initial navigation state
int32_t color_count = 0;                                            // Initial orange color count
int16_t obstacle_free_confidence = 0;                               // Initial confidence that path ahead is clear
float oa_color_count_frac = 0.18f;                                  // Threshold for obstacle detection
const int16_t max_trajectory_confidence = 5;                        // Maximum confidence value
float maxheading_increment = 10.f;                                      // Heading angle increment in degrees
float heading_increment = 10.f;
float maxDistance = 3.f;                                          // Maximum waypoint displacement in meters

// ABI binding
#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
  #define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif
static abi_event color_detection_ev;

/* ============================ */
/*    Helper Function Bodies    */
/* ============================ */

/**
 * @brief Calculates new coordinates based on current heading and distance.
 */
static void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading = stateGetNedToBodyEulers_f()->psi;
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * distanceMeters);
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * distanceMeters);

  // VERBOSE_PRINT("Calculated %f m forward position. x: %f  y: %f based on pos(%f, %f) and heading(%f)\n", distanceMeters,
  //               POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y),
  //               stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y, DegOfRad(heading));
}

/**
 * @brief Moves a waypoint to given coordinates.
 */
static void moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  // VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
  //               POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
}

/**
 * @brief Moves a waypoint forward by a certain distance.
 */
static void moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
}

/**
 * @brief Adjusts navigation heading by a given increment.
 */
static void increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
  // VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
}

/**
 * @brief Randomly chooses avoidance increment, either a positive (clockwise) or negative (counter-clockwise) increment.
 */
static void chooseRandomIncrementAvoidance(void)
{
  heading_increment = maxheading_increment * ((rand() % 100 < 70) ? 1.f : -1.f);
  // VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
}

/* ================================= */
/*    ABI Callback & Main Logic      */
/* ================================= */

/**
 * @brief Callback function triggered on visual detection events.
 *
 * Uses only the quality (orange pixel count). Other parameters are ignored.
 */
static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t __attribute__((unused)) pixel_x,
                               int16_t __attribute__((unused)) pixel_y,
                               int16_t __attribute__((unused)) pixel_width,
                               int16_t __attribute__((unused)) pixel_height,
                               int32_t quality,
                               int16_t __attribute__((unused)) extra)
{
  color_count = quality;
}

// Printing State helper funciton
void print_state(enum navigation_state_t state)
{
    switch (state) {
        case SAFE:
            printf("SAFE\n");
            break;

        case CAUTION:
            printf("CAUTION\n");
            break;

        case OBSTACLE_FOUND:
            printf("OBSTACLE_FOUND\n");
            break;

        case SEARCH_FOR_SAFE_HEADING:
            printf("SEARCH_FOR_SAFE_HEADING\n");
            break;

        case OUT_OF_BOUNDS:
            printf("OUT_OF_BOUNDS\n");
            break;

        default:
            printf("UNKNOWN STATE\n");
    }
}

/**
 * @brief Initializes the orange avoider module.
 */
void orange_avoider_init(void)
{
  srand(time(NULL));
  chooseRandomIncrementAvoidance();

  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID,
                             &color_detection_ev,
                             color_detection_cb);
}

/**
 * @brief Main (periodic) update function for obstacle avoidance.
 */
void orange_avoider_periodic(void)
{
  if (!autopilot_in_flight()) {
    return;
  }

  // Compute threshold for color detection
  int32_t color_count_threshold = oa_color_count_frac 
                                * front_camera.output_size.w 
                                * front_camera.output_size.h;

  // VERBOSE_PRINT("Color Count: %d  Threshold: %d State: %d \n", 
  //                color_count, color_count_threshold, navigation_state);
  
  // Update obstacle-free confidence
  if(color_count < color_count_threshold){
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2; // Decrease confidence if obstacle detected (the higher the decrease, 
                                   // the more sensitive the system is to obstacles)
  }
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  // Calculate the distance to move based on confidence
  // float moveDist = fminf(maxDistance, 0.2f * obstacle_free_confidence);
  float moveDist = maxDistance;

  // State machine for navigation
  print_state(navigation_state);
  switch (navigation_state){
    case SAFE: {
      // Advance trajectory waypoint based on confidence
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist);

      // Check 1: Is the new trajectory waypoint outside the obstacle zone?
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),
                              WaypointY(WP_TRAJECTORY)))
      {
        // Outside zone, transition to recovery state
        navigation_state = OUT_OF_BOUNDS;
      }
      // Check 2: If inside zone, is obstacle detection confidence zero?
      // else if (obstacle_free_confidence == 0) {
      //   // Confidence is zero (obstacle detected), transition to avoidance state
      //   navigation_state = OBSTACLE_FOUND;
      // }

      else if (obstacle_free_confidence < 0.5f) {
        // Confidence is not zero but less than 1, transition to caution state
        navigation_state = CAUTION;
      }

      // If inside zone and confidence > 0 (path appears clear):
      else {
        // Continue advancing GOAL and RETREAT waypoints.
        moveWaypointForward(WP_GOAL,    moveDist);
        moveWaypointForward(WP_RETREAT, -moveDist); // Retreat moves backward
      }
      break;
    }

    case CAUTION: {
      // Choose a random direction (clockwise or counter-clockwise) for the heading search
      increase_nav_heading(heading_increment * 10.f);
      
      // Slows down and apply a small turning
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist/2.f);
      moveWaypointForward(WP_GOAL,    moveDist/2.f);
      moveWaypointForward(WP_RETREAT, -moveDist/2.f); // Retreat moves backward



      // Set state to unsafe if the confidence drop
      if (obstacle_free_confidence == 0) {
        // Confidence is zero (obstacle detected), transition to avoidance state
        chooseRandomIncrementAvoidance();
        navigation_state = OBSTACLE_FOUND;
      }
      else if (obstacle_free_confidence >= 0.5){
        navigation_state = SAFE;
      }
      break;
    }

    case OBSTACLE_FOUND: {
      // Stop movement by setting all navigation waypoints to the current position
      // Slows down and apply a small turning
      increase_nav_heading(heading_increment * 45.f);
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist/8.f);
      moveWaypointForward(WP_GOAL,    moveDist/8.f); // Slowly backs up
      moveWaypointForward(WP_RETREAT, -moveDist/8.f); 

      // Choose a random direction (clockwise or counter-clockwise) for the heading search
      // chooseRandomIncrementAvoidance();

      // Transition to the state where the drone rotates to find a safe direction
      if (obstacle_free_confidence >= 0.25f){
        navigation_state = CAUTION;
      }
      break;
    }

    // case SEARCH_FOR_SAFE_HEADING: {
    //   // Increment the navigation heading by the chosen avoidance increment
    //   increase_nav_heading(heading_increment);

    //   // Check if obstacle-free confidence is high enough to consider the current heading safe 
    //   // If yes, transition back to the SAFE state to resume forward movement
    //   if (obstacle_free_confidence >= 2){
    //     navigation_state = SAFE;
    //   }
    //   break;
    // }

    case OUT_OF_BOUNDS: {
      // Continue rotating and moving waypoints to navigate back towards the zone
      increase_nav_heading(30.0f * heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist);
      moveWaypointForward(WP_GOAL,    moveDist/2);
      moveWaypointForward(WP_RETREAT, -moveDist/2); // Retreat moves backward

      // Check if the trajectory waypoint is now inside the obstacle zone
      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
        // Add an additional heading change to help point back into the arena
        increase_nav_heading(heading_increment);

        // Reset confidence as the safety of the path ahead from this new position is unknown
        obstacle_free_confidence = 5;

        // Ensure the direction is safe before resuming normal movement
        navigation_state = CAUTION;
      }
      break;
    }

    default:
      // Default case to catch any unexpected navigation_state values. No action is taken, just break out of the switch.
      break;
  }
}
