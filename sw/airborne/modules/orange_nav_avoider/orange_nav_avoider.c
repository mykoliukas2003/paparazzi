/*
 * Orange Nav Avoider Module
 * Combines orange pixel-based obstacle detection from orange_avoider
 * with trajectory following (circle, figure-eight, lawnmower) from depth_nav_avoider.
 */

#include "modules/orange_nav_avoider/orange_nav_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <math.h>

#include "generated/flight_plan.h"

#define ORANGE_NAV_VERBOSE TRUE
#define PRINT(string,...) fprintf(stderr, "[orange_nav_avoider->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if ORANGE_NAV_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

/* ========================= */
/*    State Machine Enums    */
/* ========================= */

enum navigation_state_t {
  SAFE,
  CAUTION,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};

enum trajectory_type_t {
  TRAJ_CIRCLE,
  TRAJ_FIGURE_EIGHT,
  TRAJ_LAWNMOWER
};

/* ================================ */
/*    Tunable Parameters            */
/* ================================ */

float oa_color_count_frac   = 0.18f;
float maxDistance           = 1.5f;
float maxheading_increment  = 10.f;
float traj_circle_radius    = 1.8f;
float traj_eight_scale      = 1.6f;
float traj_lawn_step        = 0.5f;

const int16_t max_trajectory_confidence = 5;

#define TRAJ_SNAP_DISTANCE 0.5f

/* ================================ */
/*    Global State Variables        */
/* ================================ */

enum navigation_state_t navigation_state  = SEARCH_FOR_SAFE_HEADING;
enum trajectory_type_t  active_trajectory = TRAJ_FIGURE_EIGHT;

int32_t color_count              = 0;
int16_t obstacle_free_confidence = 0;
float   heading_increment        = 10.f;

static float traj_angle     = 0.0f;
static int   lawn_direction = 1;
static float lawn_y_offset  = 0.0f;
static float arena_centre_x = 0.0f;
static float arena_centre_y = 0.0f;

/* ================================ */
/*    ABI Event                     */
/* ================================ */

#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif
static abi_event color_detection_ev;

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

/* ================================ */
/*    Function Declarations         */
/* ================================ */

static void moveWaypointForward(uint8_t waypoint, float distanceMeters);
static void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static void moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static void moveWaypointXY(uint8_t waypoint, float x, float y);
static void increase_nav_heading(float incrementDegrees);
static void set_nav_heading_towards(float tx, float ty);
static void chooseRandomIncrementAvoidance(void);
static void updateTrajectoryWaypoint(void);

/* ================================ */
/*    Init                          */
/* ================================ */

void orange_nav_avoider_init(void)
{
  srand(time(NULL));
  chooseRandomIncrementAvoidance();

  arena_centre_x = WaypointX(WP_CENTER);
  arena_centre_y = WaypointY(WP_CENTER);

  traj_angle     = 0.0f;
  lawn_y_offset  = -traj_eight_scale;
  lawn_direction = 1;

  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID,
                             &color_detection_ev,
                             color_detection_cb);

  VERBOSE_PRINT("Init. Arena centre: (%.2f, %.2f)\n",
                arena_centre_x, arena_centre_y);
}

/* ================================ */
/*    Main Periodic Function        */
/* ================================ */

void orange_nav_avoider_periodic(void)
{
  if (!autopilot_in_flight()) {
    return;
  }

  // Compute orange pixel threshold
  int32_t color_count_threshold = oa_color_count_frac
                                * front_camera.output_size.w
                                * front_camera.output_size.h;

  // Update confidence
  if (color_count < color_count_threshold) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  VERBOSE_PRINT("ColorCount:%d Threshold:%d Conf:%d State:%d Traj:%d\n",
                color_count, color_count_threshold,
                obstacle_free_confidence, navigation_state, active_trajectory);

  float moveDist = maxDistance;

  switch (navigation_state) {

    case SAFE: {
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist);

      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),
                              WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
        break;
      }

      if (obstacle_free_confidence < 1) {
        navigation_state = CAUTION;
        VERBOSE_PRINT("Entering CAUTION\n");
        break;
      }

      // Path clear — follow trajectory
      updateTrajectoryWaypoint();
      moveWaypointForward(WP_RETREAT, -moveDist);
      break;
    }

    case CAUTION: {
      increase_nav_heading(heading_increment * 10.f);
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist / 2.f);
      moveWaypointForward(WP_GOAL,       moveDist / 2.f);
      moveWaypointForward(WP_RETREAT,   -moveDist / 2.f);

      if (obstacle_free_confidence == 0) {
        chooseRandomIncrementAvoidance();
        navigation_state = OBSTACLE_FOUND;
        VERBOSE_PRINT("Entering OBSTACLE_FOUND\n");
      } else if (obstacle_free_confidence >= 2) {
        navigation_state = SAFE;
        // Snap trajectory angle to current position for smooth rejoin
        float cur_x = stateGetPositionEnu_f()->x;
        float cur_y = stateGetPositionEnu_f()->y;
        traj_angle = atan2f(cur_y - arena_centre_y,
                            cur_x - arena_centre_x);
      }
      break;
    }

    case OBSTACLE_FOUND: {
      increase_nav_heading(heading_increment * 45.f);
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist / 8.f);
      moveWaypointForward(WP_GOAL,       moveDist / 8.f);
      moveWaypointForward(WP_RETREAT,   -moveDist / 8.f);

      if (obstacle_free_confidence >= 1) {
        navigation_state = CAUTION;
        VERBOSE_PRINT("Obstacle clearing, entering CAUTION\n");
      }
      break;
    }

    case SEARCH_FOR_SAFE_HEADING: {
      increase_nav_heading(heading_increment);

      if (obstacle_free_confidence >= 2) {
        VERBOSE_PRINT("Safe heading found, resuming SAFE\n");
        navigation_state = SAFE;
        traj_angle = atan2f(stateGetPositionEnu_f()->y - arena_centre_y,
                            stateGetPositionEnu_f()->x - arena_centre_x);
      }
      break;
    }

    case OUT_OF_BOUNDS: {
      static uint8_t oob_cycles = 0;
      oob_cycles++;

      increase_nav_heading(30.0f * heading_increment / fabsf(heading_increment));
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist);
      moveWaypointForward(WP_GOAL,       moveDist / 2.f);
      moveWaypointForward(WP_RETREAT,   -moveDist / 2.f);

      if (oob_cycles >= 3 &&
          InsideObstacleZone(WaypointX(WP_TRAJECTORY),
                             WaypointY(WP_TRAJECTORY))) {
        oob_cycles = 0;
        obstacle_free_confidence = 2;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
        VERBOSE_PRINT("Back inside zone\n");
      }
      break;
    }

    default:
      break;
  }
}

/* ================================ */
/*    Trajectory Logic              */
/* ================================ */

static void updateTrajectoryWaypoint(void)
{
  float target_x, target_y;

  switch (active_trajectory) {

    case TRAJ_CIRCLE: {
      traj_angle += 0.05f;
      FLOAT_ANGLE_NORMALIZE(traj_angle);
      target_x = arena_centre_x + traj_circle_radius * cosf(traj_angle);
      target_y = arena_centre_y + traj_circle_radius * sinf(traj_angle);
      VERBOSE_PRINT("CIRCLE target: (%.2f, %.2f)\n", target_x, target_y);
      break;
    }

    case TRAJ_FIGURE_EIGHT: {
      traj_angle += 0.05f;
      FLOAT_ANGLE_NORMALIZE(traj_angle);
      float denom = 1.0f + sinf(traj_angle) * sinf(traj_angle);
      target_x = arena_centre_x + traj_eight_scale * cosf(traj_angle) / denom;
      target_y = arena_centre_y + traj_eight_scale * sinf(traj_angle) * cosf(traj_angle) / denom;
      VERBOSE_PRINT("FIGURE-EIGHT target: (%.2f, %.2f)\n", target_x, target_y);
      break;
    }

    case TRAJ_LAWNMOWER: {
      float cur_x = stateGetPositionEnu_f()->x;
      float cur_y = stateGetPositionEnu_f()->y;
      float sweep_end_x = arena_centre_x + lawn_direction * traj_eight_scale;
      float sweep_end_y = arena_centre_y + lawn_y_offset;
      float dx = sweep_end_x - cur_x;
      float dy = sweep_end_y - cur_y;

      if (sqrtf(dx*dx + dy*dy) < TRAJ_SNAP_DISTANCE) {
        lawn_direction *= -1;
        lawn_y_offset  += traj_lawn_step * lawn_direction;
        if (lawn_y_offset >  traj_eight_scale) lawn_y_offset =  traj_eight_scale;
        if (lawn_y_offset < -traj_eight_scale) lawn_y_offset = -traj_eight_scale;
        VERBOSE_PRINT("LAWNMOWER: reversing, y_offset: %.2f\n", lawn_y_offset);
      }

      target_x = arena_centre_x + lawn_direction * traj_eight_scale;
      target_y = arena_centre_y + lawn_y_offset;
      VERBOSE_PRINT("LAWNMOWER target: (%.2f, %.2f)\n", target_x, target_y);
      break;
    }

    default:
      moveWaypointForward(WP_GOAL, maxDistance);
      return;
  }

  moveWaypointXY(WP_GOAL, target_x, target_y);
  set_nav_heading_towards(target_x, target_y);
}

/* ================================ */
/*    Public Trajectory Selector    */
/* ================================ */

void orange_nav_avoider_trajectory(uint8_t traj_id)
{
  switch (traj_id) {
    case 0:  active_trajectory = TRAJ_CIRCLE;       break;
    case 1:  active_trajectory = TRAJ_FIGURE_EIGHT; break;
    case 2:  active_trajectory = TRAJ_LAWNMOWER;    break;
    default: active_trajectory = TRAJ_FIGURE_EIGHT; break;
  }
  traj_angle     = 0.0f;
  lawn_y_offset  = -traj_eight_scale;
  lawn_direction = 1;
  VERBOSE_PRINT("Trajectory set to: %d\n", active_trajectory);
}

/* ================================ */
/*    Helper Functions              */
/* ================================ */

static void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading = stateGetNedToBodyEulers_f()->psi;
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * distanceMeters);
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * distanceMeters);
}

static void moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
}

static void moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
}

static void moveWaypointXY(uint8_t waypoint, float x, float y)
{
  struct EnuCoor_i new_coor;
  new_coor.x = POS_BFP_OF_REAL(x);
  new_coor.y = POS_BFP_OF_REAL(y);
  new_coor.z = stateGetPositionEnu_i()->z;
  waypoint_move_xy_i(waypoint, new_coor.x, new_coor.y);
}

static void increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
}

static void set_nav_heading_towards(float tx, float ty)
{
  float cur_x = stateGetPositionEnu_f()->x;
  float cur_y = stateGetPositionEnu_f()->y;
  float desired_heading = atan2f(tx - cur_x, ty - cur_y);
  FLOAT_ANGLE_NORMALIZE(desired_heading);
  nav.heading = desired_heading;
}

static void chooseRandomIncrementAvoidance(void)
{
  heading_increment = maxheading_increment * ((rand() % 100 < 70) ? 1.f : -1.f);
  VERBOSE_PRINT("Heading increment: %.1f\n", heading_increment);
}