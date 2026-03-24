/**
 * @file "modules/cnn_avoid/cnn_avoid.c"
 *
 * CNN-depth based obstacle avoidance with trajectory following.
 *
 * Reads the 7-block depth vector produced by cnn_vision directly
 * (shared global, no ABI message needed).
 *
 * Block layout (left-to-right in the image):
 *   [0] [1]  |  [2] [3] [4]  |  [5] [6]
 *     LEFT       STRAIGHT        RIGHT
 *
 * Each block holds the maximum depth in that vertical strip.
 * Higher value = more free space (obstacle further away).
 * We take the MINIMUM within each region so the closest obstacle dominates.
 *
 * Key features:
 *  - Directional turning based on left vs right depth
 *  - Confidence system to avoid false positives
 *  - Low-pass filter on straight depth to reduce noise
 *  - CAUTION state with hysteresis
 *  - Emergency raw-depth override
 *  - Trajectory following: circle, figure-eight, lawnmower
 *  - Obstacle-arc skipping on trajectory rejoin
 *  - OUT_OF_BOUNDS turns toward arena centre
 */

#include "modules/cnn_avoid/cnn_avoid.h"
#include "modules/computer_vision/cnn_vision/cnn_vision.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include <time.h>
#include <stdio.h>
#include <math.h>
#include "generated/flight_plan.h"

/* ===================== */
/*    Debug Printing     */
/* ===================== */

#define CNN_AVOID_VERBOSE TRUE
#define PRINT(string,...) fprintf(stderr, "[cnn_avoid->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if CNN_AVOID_VERBOSE
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

/*
 * Depth thresholds.
 * The CNN outputs values in a model-specific range (typically [0, 255]).
 * Higher = more free space.  Tune these to your model's output scale.
 */
float cnn_safe_distance_threshold    = 0.48f;   // hard stop
float cnn_caution_distance_threshold = 0.53f;   // slow down
float cnn_caution_exit_threshold     = 0.57f;   // hysteresis exit

/* Low-pass filter on straight depth (0 = no filter, 1 = frozen) */
float cnn_depth_filter_alpha = 0.7f;

float cnn_max_distance = 1.0f;  // max forward displacement per cycle [m]

const int16_t max_trajectory_confidence = 3;

/* Trajectory parameters */
float traj_circle_radius = 1.8f;
float traj_circle_speed  = 0.05f;
float traj_eight_scale   = 1.6f;
float traj_lawn_step     = 0.5f;

#define TRAJ_SNAP_DISTANCE 0.5f

/* ================================ */
/*    Global State Variables        */
/* ================================ */

enum navigation_state_t navigation_state  = SEARCH_FOR_SAFE_HEADING;
enum trajectory_type_t  active_trajectory = TRAJ_FIGURE_EIGHT;

int16_t obstacle_free_confidence = 0;
float   heading_increment        = 15.0f;

/* Derived from the 7-block vector each cycle */
static float depth_left     = 0.0f;
static float depth_straight = 0.0f;
static float depth_right    = 0.0f;
static float depth_straight_filtered = 0.0f;

/* Trajectory state */
static float traj_angle     = 0.0f;
static int   lawn_direction = 1;
static float lawn_y_offset  = 0.0f;

/* Obstacle-arc memory */
static float obstacle_angle = 0.0f;

/* Arena centre (ENU, from WP_CENTER) */
static float arena_centre_x = 0.0f;
static float arena_centre_y = 0.0f;

static bool trajectory_active = false;

/* ================================ */
/*    Function Declarations         */
/* ================================ */

static void moveWaypointForward(uint8_t waypoint, float distanceMeters);
static void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static void moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static void moveWaypointXY(uint8_t waypoint, float x, float y);
static void increase_nav_heading(float incrementDegrees);
static void set_nav_heading_towards(float tx, float ty);
static void chooseDirectionalIncrement(void);
static void updateTrajectoryWaypoint(void);
static void update_depths_from_nav_vector(void);

/* ================================ */
/*    Init                          */
/* ================================ */

void cnn_avoid_init(void)
{
  srand(time(NULL));
  chooseDirectionalIncrement();

  depth_straight_filtered = cnn_caution_distance_threshold;

  arena_centre_x = WaypointX(WP_CENTER);
  arena_centre_y = WaypointY(WP_CENTER);

  traj_angle     = 0.0f;
  obstacle_angle = 0.0f;
  lawn_y_offset  = -traj_eight_scale;

  VERBOSE_PRINT("Init complete. Arena centre: (%.2f, %.2f)\n",
                arena_centre_x, arena_centre_y);
}

/* ============================================= */
/*    Map 7-block vector to left/straight/right  */
/* ============================================= */

/**
 * @brief Reads cnn_vision_nav_vector[0..6] and computes
 *        depth_left, depth_straight, depth_right.
 *
 * Uses the MINIMUM within each region so the closest
 * obstacle in that direction dominates.
 */
static void update_depths_from_nav_vector(void)
{
  if (!cnn_vision_nav_valid) {
    /* No data yet — keep previous values */
    return;
  }

  const float *v = cnn_vision_nav_vector;

  /* Left: blocks 0-1 */
  depth_left = v[0];
  if (v[1] < depth_left) depth_left = v[1];

  /* Straight: blocks 2-4 */
  depth_straight = v[2];
  if (v[3] < depth_straight) depth_straight = v[3];
  if (v[4] < depth_straight) depth_straight = v[4];

  /* Right: blocks 5-6 */
  depth_right = v[5];
  if (v[6] < depth_right) depth_right = v[6];
}

/* ================================ */
/*    Main Periodic Function        */
/* ================================ */

void cnn_avoid_periodic(void)
{
  if (!autopilot_in_flight()) return;

  /* ---- Read latest depths from vision module ---- */
  update_depths_from_nav_vector();

  /* ---- Low-pass filter on straight depth ---- */
  depth_straight_filtered = cnn_depth_filter_alpha * depth_straight
                          + (1.0f - cnn_depth_filter_alpha) * depth_straight_filtered;

  VERBOSE_PRINT("Depths L:%.2f S:%.2f(f:%.2f) R:%.2f | Conf:%d | State:%d | Traj:%d\n",
                depth_left, depth_straight, depth_straight_filtered, depth_right,
                obstacle_free_confidence, navigation_state, active_trajectory);

  float moveDist = cnn_max_distance;

  /* ---- Update confidence ---- */
  if (depth_straight_filtered >= cnn_caution_distance_threshold) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  /* ---- Emergency override: raw depth critical while in SAFE ---- */
  if (navigation_state == SAFE && depth_straight < cnn_safe_distance_threshold) {
    VERBOSE_PRINT("EMERGENCY: raw depth %.2f below safe threshold\n", depth_straight);
    obstacle_angle = traj_angle;
    if (depth_left > depth_right) {
      heading_increment = -15.0f;
    } else {
      heading_increment =  15.0f;
    }
    obstacle_free_confidence = 0;
    navigation_state = OBSTACLE_FOUND;
    trajectory_active = false;
  }

  /* ---- State Machine ---- */
  switch (navigation_state) {

    case SAFE: {
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist);

      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),
                              WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
        trajectory_active = false;
        VERBOSE_PRINT("Out of bounds!\n");
        break;
      }

      if (depth_straight_filtered < cnn_caution_distance_threshold) {
        navigation_state = CAUTION;
        trajectory_active = false;
        VERBOSE_PRINT("Entering CAUTION, filtered depth: %.2f\n",
                      depth_straight_filtered);
        break;
      }

      trajectory_active = true;
      updateTrajectoryWaypoint();
      moveWaypointForward(WP_RETREAT, -moveDist);
      break;
    }

    case CAUTION: {
      chooseDirectionalIncrement();
      increase_nav_heading(heading_increment * 0.5f);
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist * 0.5f);
      moveWaypointForward(WP_GOAL,       moveDist * 0.5f);
      moveWaypointForward(WP_RETREAT,   -moveDist * 0.5f);

      if (depth_straight_filtered < cnn_safe_distance_threshold) {
        obstacle_angle = traj_angle;
        if (depth_left > depth_right) {
          heading_increment = -15.0f;
          VERBOSE_PRINT("More space LEFT (L:%.2f > R:%.2f)\n", depth_left, depth_right);
        } else {
          heading_increment =  15.0f;
          VERBOSE_PRINT("More space RIGHT (R:%.2f >= L:%.2f)\n", depth_right, depth_left);
        }
        obstacle_free_confidence = 0;
        navigation_state = OBSTACLE_FOUND;
        VERBOSE_PRINT("Escalating to OBSTACLE_FOUND\n");
        break;
      }

      if (depth_straight_filtered >= cnn_caution_exit_threshold &&
          obstacle_free_confidence >= 2) {
        navigation_state = SAFE;
        VERBOSE_PRINT("CAUTION resolved\n");
      }
      break;
    }

    case OBSTACLE_FOUND: {
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);
      moveWaypointForward(WP_RETREAT, -moveDist * 0.25f);

      increase_nav_heading(heading_increment);

      if (depth_straight_filtered >= cnn_safe_distance_threshold &&
          obstacle_free_confidence >= 1) {
        navigation_state = SEARCH_FOR_SAFE_HEADING;
        VERBOSE_PRINT("Obstacle cleared, searching for safe heading\n");
      }
      break;
    }

    case SEARCH_FOR_SAFE_HEADING: {
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);
      increase_nav_heading(heading_increment);

      if (depth_straight_filtered >= cnn_caution_distance_threshold &&
          obstacle_free_confidence >= 2) {
        VERBOSE_PRINT("Safe heading found, resuming SAFE\n");
        navigation_state = SAFE;

        float cur_x = stateGetPositionEnu_f()->x;
        float cur_y = stateGetPositionEnu_f()->y;

        if (active_trajectory == TRAJ_CIRCLE || active_trajectory == TRAJ_FIGURE_EIGHT) {
          float cur_angle  = atan2f(cur_y - arena_centre_y, cur_x - arena_centre_x);
          float skip_angle = obstacle_angle + 1.5f;
          FLOAT_ANGLE_NORMALIZE(skip_angle);

          float dist_cur  = cur_angle  - obstacle_angle;
          float dist_skip = skip_angle - obstacle_angle;
          if (dist_cur  < 0) dist_cur  += 2.0f * M_PI;
          if (dist_skip < 0) dist_skip += 2.0f * M_PI;

          traj_angle = (dist_cur > dist_skip)
                     ? cur_angle + 0.3f
                     : skip_angle;
          FLOAT_ANGLE_NORMALIZE(traj_angle);

          VERBOSE_PRINT("Rejoin: obstacle=%.2f skip=%.2f traj=%.2f\n",
                        obstacle_angle, skip_angle, traj_angle);
        }
        else if (active_trajectory == TRAJ_LAWNMOWER) {
          lawn_y_offset += traj_lawn_step * (float)lawn_direction;
          lawn_direction *= -1;
          if (lawn_y_offset >  traj_eight_scale) lawn_y_offset =  traj_eight_scale;
          if (lawn_y_offset < -traj_eight_scale) lawn_y_offset = -traj_eight_scale;
        }
      }
      break;
    }

    case OUT_OF_BOUNDS: {
      static uint8_t oob_cycles = 0;
      oob_cycles++;

      set_nav_heading_towards(arena_centre_x, arena_centre_y);
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist);
      moveWaypointForward(WP_GOAL,       moveDist * 0.5f);
      moveWaypointForward(WP_RETREAT,   -moveDist * 0.5f);

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
      traj_angle += traj_circle_speed;
      FLOAT_ANGLE_NORMALIZE(traj_angle);
      target_x = arena_centre_x + traj_circle_radius * cosf(traj_angle);
      target_y = arena_centre_y + traj_circle_radius * sinf(traj_angle);
      VERBOSE_PRINT("CIRCLE target: (%.2f, %.2f) angle: %.2f\n",
                    target_x, target_y, traj_angle);
      break;
    }

    case TRAJ_FIGURE_EIGHT: {
      traj_angle += traj_circle_speed;
      FLOAT_ANGLE_NORMALIZE(traj_angle);
      float denom = 1.0f + sinf(traj_angle) * sinf(traj_angle);
      target_x = arena_centre_x + traj_eight_scale * cosf(traj_angle) / denom;
      target_y = arena_centre_y + traj_eight_scale * sinf(traj_angle) * cosf(traj_angle) / denom;
      VERBOSE_PRINT("EIGHT target: (%.2f, %.2f) angle: %.2f\n",
                    target_x, target_y, traj_angle);
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
        lawn_y_offset += traj_lawn_step * (float)lawn_direction;
        lawn_direction *= -1;
        if (lawn_y_offset >  traj_eight_scale) lawn_y_offset =  traj_eight_scale;
        if (lawn_y_offset < -traj_eight_scale) lawn_y_offset = -traj_eight_scale;
        VERBOSE_PRINT("LAWNMOWER: reversed, y_offset: %.2f\n", lawn_y_offset);
      }

      target_x = sweep_end_x;
      target_y = sweep_end_y;
      VERBOSE_PRINT("LAWNMOWER target: (%.2f, %.2f)\n", target_x, target_y);
      break;
    }

    default:
      moveWaypointForward(WP_GOAL, cnn_max_distance);
      return;
  }

  moveWaypointXY(WP_GOAL, target_x, target_y);
  set_nav_heading_towards(target_x, target_y);
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
  float desired = atan2f(tx - cur_x, ty - cur_y);
  FLOAT_ANGLE_NORMALIZE(desired);
  nav.heading = desired;
}

/**
 * @brief Pick turn direction based on which side has more free space.
 *        Higher depth = more space.
 */
static void chooseDirectionalIncrement(void)
{
  if (depth_left > depth_right) {
    heading_increment = -5.0f;
  } else if (depth_right > depth_left) {
    heading_increment =  5.0f;
  } else {
    heading_increment = (rand() % 2 == 0) ? 5.0f : -5.0f;
  }
  VERBOSE_PRINT("chooseDir: L:%.2f R:%.2f → increment=%.1f\n",
                depth_left, depth_right, heading_increment);
}

/* ================================ */
/*    Public Trajectory Selector    */
/* ================================ */

void cnn_avoid_set_trajectory(uint8_t traj_id)
{
  switch (traj_id) {
    case 0: active_trajectory = TRAJ_CIRCLE;       break;
    case 1: active_trajectory = TRAJ_FIGURE_EIGHT; break;
    case 2: active_trajectory = TRAJ_LAWNMOWER;    break;
    default: active_trajectory = TRAJ_FIGURE_EIGHT; break;
  }
  traj_angle     = 0.0f;
  obstacle_angle = 0.0f;
  lawn_y_offset  = -traj_eight_scale;
  lawn_direction = 1;
  VERBOSE_PRINT("Trajectory set to: %d\n", active_trajectory);
}