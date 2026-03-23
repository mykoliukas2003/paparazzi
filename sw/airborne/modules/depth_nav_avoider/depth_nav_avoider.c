/*
 * Depth Nav Avoider Module
 * Merges the confidence-based state machine from orange_avoider
 * with the depth-aware, directional turning from depth_avoider.
 * Adds trajectory following: circle, figure-eight, and lawnmower.
 *
 * Key features:
 *  - Uses depth (left/straight/right) for smarter turn decisions
 *  - Confidence system to avoid false positives
 *  - Low-pass filter on depth_straight to reduce noise
 *  - CAUTION state for gradual obstacle response
 *  - RETREAT waypoint for backing away from obstacles
 *  - Trajectory following with return-to-path after avoidance
 *  - OUT_OF_BOUNDS minimum cycle counter to prevent thrashing
 */

#include "modules/depth_nav_avoider/depth_nav_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <math.h>
#include "generated/flight_plan.h"

/* ===================== */
/*    Debug Printing     */
/* ===================== */

#define DEPTH_NAV_AVOIDER_VERBOSE TRUE
#define PRINT(string,...) fprintf(stderr, "[depth_nav_avoider->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if DEPTH_NAV_AVOIDER_VERBOSE
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
  TRAJ_CIRCLE,        // Circular loop around arena centre
  TRAJ_FIGURE_EIGHT,  // Figure-eight through arena centre
  TRAJ_LAWNMOWER      // Diagonal sweep corner to corner
};

/* ================================ */
/*    Tunable Parameters            */
/* ================================ */

// Depth thresholds (normalized depth scale: approx -32 to +32)
float safe_distance_threshold    = 5.0f;   // hard stop threshold
float caution_distance_threshold = 10.0f;  // early warning threshold

// Depth noise filter (0 = no filtering, 1 = never updates)
// 0.4 means 40% new reading, 60% history — good balance for this sensor
#define DEPTH_FILTER_ALPHA 0.4f

// Max forward displacement per cycle [m]
float maxDistance = 1.0f;

// Confidence system: higher = less sensitive to brief obstacle detections
const int16_t max_trajectory_confidence = 3;

// Trajectory parameters
float traj_circle_radius  = 1.8f;   // metres — sized for ~5x5m CyberZoo
float traj_circle_speed   = 0.05f;  // radians per cycle along circle
float traj_eight_scale    = 1.6f;   // half-width of figure-eight [m]
float traj_lawn_step      = 0.5f;   // lateral step between lawnmower passes [m]

// How close drone must be to trajectory before "on path" [m]
#define TRAJ_SNAP_DISTANCE 0.5f

/* ================================ */
/*    Global State Variables        */
/* ================================ */

enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
enum trajectory_type_t  active_trajectory = TRAJ_FIGURE_EIGHT;

int16_t obstacle_free_confidence = 0;
float   heading_increment        = 5.f;

// Depth readings from ABI
float depth_left             = 0.0f;
float depth_straight         = 0.0f;
float depth_right            = 0.0f;
static float depth_straight_filtered = 0.0f;

// Trajectory state
static float traj_angle      = 0.0f;   // current angle along circle/eight [rad]
static int   lawn_direction  = 1;      // +1 or -1 for lawnmower sweep direction
static float lawn_y_offset   = 0.0f;  // current lawnmower lateral position [m]

// Arena centre (ENU coordinates, set at init from WP_CENTER)
static float arena_centre_x  = 0.0f;
static float arena_centre_y  = 0.0f;

// Flag: are we currently in trajectory-following mode or pure avoidance
static bool trajectory_active = false;

/* ================================ */
/*    ABI Event                     */
/* ================================ */

#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif
static abi_event depth_detection_ev;

/**
 * @brief ABI callback — receives depth estimates packed as scaled integers.
 * cv_estimate_depth packs floats*1000 into pixel_x, pixel_y, pixel_width.
 */
static void depth_detection_cb(uint8_t __attribute__((unused)) sender_id,
                                int16_t pixel_x,
                                int16_t pixel_y,
                                int16_t pixel_width,
                                int16_t __attribute__((unused)) pixel_height,
                                int32_t __attribute__((unused)) quality,
                                int16_t __attribute__((unused)) extra)
{
  depth_left     = ((float)pixel_x)     / 100.0f;
  depth_straight = ((float)pixel_y)     / 100.0f;
  depth_right    = ((float)pixel_width) / 100.0f;
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
static void chooseDirectionalIncrement(void);
static void updateTrajectoryWaypoint(void);
static float distanceToTrajectory(float px, float py);

/* ================================ */
/*    Init                          */
/* ================================ */

/**
 * @brief Initialises the depth nav avoider module.
 */
void depth_nav_avoider_init(void)
{
  srand(time(NULL));
  chooseDirectionalIncrement();

  // Store arena centre from WP_CENTER waypoint defined in flight plan
  arena_centre_x = WaypointX(WP_CENTER);
  arena_centre_y = WaypointY(WP_CENTER);

  // Start trajectory angle pointing forward (north)
  traj_angle = 0.0f;
  lawn_y_offset = -traj_eight_scale; // start lawnmower at one edge

  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID,
                             &depth_detection_ev,
                             depth_detection_cb);

  VERBOSE_PRINT("Init complete. Arena centre: (%.2f, %.2f)\n",
                arena_centre_x, arena_centre_y);
}

/* ================================ */
/*    Main Periodic Function        */
/* ================================ */

/**
 * @brief Main update loop. Runs trajectory following + avoidance state machine.
 */
void depth_nav_avoider_periodic(void)
{
  if (!autopilot_in_flight()) {
    return;
  }

  // ---- Low-pass filter on depth_straight to reduce noise ----
  depth_straight_filtered = DEPTH_FILTER_ALPHA * depth_straight
                          + (1.0f - DEPTH_FILTER_ALPHA) * depth_straight_filtered;

  VERBOSE_PRINT("Depths L:%.3f S:%.3f(f:%.3f) R:%.3f | Conf:%d | State:%d | Traj:%d\n",
                depth_left, depth_straight, depth_straight_filtered, depth_right,
                obstacle_free_confidence, navigation_state, active_trajectory);

  float moveDist = maxDistance;

  // ---- Update confidence based on filtered depth ----
  if (depth_straight_filtered >= caution_distance_threshold) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  // ---- State Machine ----
  switch (navigation_state) {

    case SAFE: {
      // Probe trajectory waypoint ahead
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist);

      // Check geofence first
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),
                              WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
        trajectory_active = false;
        break;
      }

      // Check for obstacle
      if (depth_straight_filtered < caution_distance_threshold) {
        navigation_state = CAUTION;
        trajectory_active = false;
        VERBOSE_PRINT("Entering CAUTION, filtered depth: %.3f\n",
                      depth_straight_filtered);
        break;
      }

      // Path is clear — follow trajectory
      trajectory_active = true;
      updateTrajectoryWaypoint();
      moveWaypointForward(WP_RETREAT, -moveDist);
      break;
    }

    case CAUTION: {
      // Slow down, apply small corrective turn
      increase_nav_heading(heading_increment * 5.0f);
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist / 2.0f);
      moveWaypointForward(WP_GOAL,       moveDist / 2.0f);
      moveWaypointForward(WP_RETREAT,   -moveDist / 2.0f);

      // Obstacle confirmed → OBSTACLE_FOUND
      if (depth_straight_filtered < safe_distance_threshold) {
        if (fabsf(depth_left) > fabsf(depth_right)) {
          heading_increment = -15.0f;
          VERBOSE_PRINT("More space LEFT, turning CCW\n");
        } else {
          heading_increment = 15.0f;
          VERBOSE_PRINT("More space RIGHT, turning CW\n");
        }
        obstacle_free_confidence = 0;
        navigation_state = OBSTACLE_FOUND;
      }
      // Path cleared → back to SAFE
      else if (depth_straight_filtered >= caution_distance_threshold &&
               obstacle_free_confidence >= 2) {
        navigation_state = SAFE;
      }
      break;
    }

    case OBSTACLE_FOUND: {
      // Freeze forward motion
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      // Back up
      moveWaypointForward(WP_RETREAT, -moveDist / 4.0f);

      // Rotate, re-evaluating direction each cycle with absolute depth comparison
      increase_nav_heading(heading_increment);
      if (fabsf(depth_left) > fabsf(depth_right)) {
        heading_increment = -15.0f;
      } else {
        heading_increment =  15.0f;
      }

      if (depth_straight_filtered >= safe_distance_threshold &&
          obstacle_free_confidence >= 1) {
        navigation_state = SEARCH_FOR_SAFE_HEADING;
        VERBOSE_PRINT("Obstacle cleared, searching for safe heading\n");
      }
      break;
    }

    case SEARCH_FOR_SAFE_HEADING: {
      increase_nav_heading(heading_increment);

      if (depth_straight_filtered >= caution_distance_threshold &&
          obstacle_free_confidence >= 2) {
        VERBOSE_PRINT("Safe heading found, resuming SAFE\n");
        navigation_state = SAFE;

        // Snap trajectory angle to nearest point so we rejoin smoothly
        float cur_x = stateGetPositionEnu_f()->x;
        float cur_y = stateGetPositionEnu_f()->y;
        if (active_trajectory == TRAJ_CIRCLE || active_trajectory == TRAJ_FIGURE_EIGHT) {
          // Reset traj_angle to point nearest to current position
          traj_angle = atan2f(cur_y - arena_centre_y, cur_x - arena_centre_x);
        }
      }
      break;
    }

    case OUT_OF_BOUNDS: {
      // Minimum cycles in OOB before checking re-entry — prevents thrashing
      static uint8_t oob_cycles = 0;
      oob_cycles++;

      increase_nav_heading(15.0f * heading_increment / fabsf(heading_increment));
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist);
      moveWaypointForward(WP_GOAL,       moveDist / 2.0f);
      moveWaypointForward(WP_RETREAT,   -moveDist / 2.0f);

      if (oob_cycles >= 3 &&
          InsideObstacleZone(WaypointX(WP_TRAJECTORY),
                             WaypointY(WP_TRAJECTORY))) {
        oob_cycles = 0;
        obstacle_free_confidence = 2;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
        VERBOSE_PRINT("Back inside zone, verifying heading\n");
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

/**
 * @brief Computes the next trajectory waypoint and steers WP_GOAL toward it.
 *        Called only when state == SAFE and path is clear.
 */
static void updateTrajectoryWaypoint(void)
{
  float target_x, target_y;

  switch (active_trajectory) {

    case TRAJ_CIRCLE: {
      // Advance angle along circle
      traj_angle += traj_circle_speed;
      FLOAT_ANGLE_NORMALIZE(traj_angle);

      target_x = arena_centre_x + traj_circle_radius * cosf(traj_angle);
      target_y = arena_centre_y + traj_circle_radius * sinf(traj_angle);

      VERBOSE_PRINT("CIRCLE target: (%.2f, %.2f) angle: %.2f\n",
                    target_x, target_y, traj_angle);
      break;
    }

    case TRAJ_FIGURE_EIGHT: {
      // Lemniscate of Bernoulli parametric form, scaled to arena
      // x = scale * cos(t) / (1 + sin²(t))
      // y = scale * sin(t)*cos(t) / (1 + sin²(t))
      traj_angle += traj_circle_speed;
      FLOAT_ANGLE_NORMALIZE(traj_angle);

      float denom = 1.0f + sinf(traj_angle) * sinf(traj_angle);
      target_x = arena_centre_x + traj_eight_scale * cosf(traj_angle) / denom;
      target_y = arena_centre_y + traj_eight_scale * sinf(traj_angle) * cosf(traj_angle) / denom;

      VERBOSE_PRINT("FIGURE-EIGHT target: (%.2f, %.2f) angle: %.2f\n",
                    target_x, target_y, traj_angle);
      break;
    }

    case TRAJ_LAWNMOWER: {
      // Sweep diagonally across arena, step laterally at each end
      float cur_x = stateGetPositionEnu_f()->x;
      float cur_y = stateGetPositionEnu_f()->y;

      // Target is always the far end of the current sweep pass
      float sweep_end_x = arena_centre_x + lawn_direction * traj_eight_scale;
      float sweep_end_y = arena_centre_y + lawn_y_offset;

      // Check if we are near the end of the current pass
      float dx = sweep_end_x - cur_x;
      float dy = sweep_end_y - cur_y;
      float dist_to_end = sqrtf(dx*dx + dy*dy);

      if (dist_to_end < TRAJ_SNAP_DISTANCE) {
        // Reached end of pass — reverse direction and step laterally
        lawn_direction *= -1;
        lawn_y_offset  += traj_lawn_step * lawn_direction;

        // Clamp to arena bounds
        if (lawn_y_offset > traj_eight_scale)  lawn_y_offset =  traj_eight_scale;
        if (lawn_y_offset < -traj_eight_scale) lawn_y_offset = -traj_eight_scale;

        VERBOSE_PRINT("LAWNMOWER: reversing, new y_offset: %.2f\n", lawn_y_offset);
      }

      target_x = sweep_end_x;
      target_y = sweep_end_y;

      VERBOSE_PRINT("LAWNMOWER target: (%.2f, %.2f)\n", target_x, target_y);
      break;
    }

    default:
      // Fallback: just move forward
      moveWaypointForward(WP_GOAL, maxDistance);
      return;
  }

  // Move WP_GOAL to trajectory target and steer heading toward it
  moveWaypointXY(WP_GOAL, target_x, target_y);
  set_nav_heading_towards(target_x, target_y);
}

/**
 * @brief Returns approximate distance from point (px,py) to the current trajectory.
 *        Used to snap traj_angle after avoidance.
 */
static float distanceToTrajectory(float px, float py)
{
  float dx = px - arena_centre_x;
  float dy = py - arena_centre_y;
  float dist = sqrtf(dx*dx + dy*dy);

  if (active_trajectory == TRAJ_CIRCLE) {
    return fabsf(dist - traj_circle_radius);
  }
  // For figure-eight and lawnmower, approximate as distance to centre
  return dist;
}

/* ================================ */
/*    Helper Function Bodies        */
/* ================================ */

/**
 * @brief Calculates new coordinates based on current heading and distance.
 */
static void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading = stateGetNedToBodyEulers_f()->psi;
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * distanceMeters);
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * distanceMeters);
}

/**
 * @brief Moves a waypoint to given coordinates.
 */
static void moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
}

/**
 * @brief Moves a waypoint forward by a certain distance in current heading.
 */
static void moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
}

/**
 * @brief Moves a waypoint to absolute ENU x/y coordinates.
 */
static void moveWaypointXY(uint8_t waypoint, float x, float y)
{
  struct EnuCoor_i new_coor;
  new_coor.x = POS_BFP_OF_REAL(x);
  new_coor.y = POS_BFP_OF_REAL(y);
  new_coor.z = stateGetPositionEnu_i()->z;
  waypoint_move_xy_i(waypoint, new_coor.x, new_coor.y);
}

/**
 * @brief Adjusts navigation heading by a given increment.
 */
static void increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
}

/**
 * @brief Points the drone's heading toward a target ENU position.
 */
static void set_nav_heading_towards(float tx, float ty)
{
  float cur_x = stateGetPositionEnu_f()->x;
  float cur_y = stateGetPositionEnu_f()->y;
  float desired_heading = atan2f(tx - cur_x, ty - cur_y);
  FLOAT_ANGLE_NORMALIZE(desired_heading);
  nav.heading = desired_heading;
}

/**
 * @brief Uses depth readings to choose the better turn direction.
 *        Uses absolute values to handle the -32 to +32 depth scale.
 *        Falls back to random if depths are equal or unavailable.
 */
static void chooseDirectionalIncrement(void)
{
  if (fabsf(depth_left) > fabsf(depth_right)) {
    heading_increment = -5.0f; // CCW (left has more open space)
  } else if (fabsf(depth_right) > fabsf(depth_left)) {
    heading_increment =  5.0f; // CW (right has more open space)
  } else {
    heading_increment = (rand() % 2 == 0) ? 5.0f : -5.0f;
  }
  VERBOSE_PRINT("Chosen heading increment: %.1f\n", heading_increment);
}

/* ================================ */
/*    Public Trajectory Selector    */
/* ================================ */

/**
 * @brief Called from flight plan to select which trajectory to follow.
 *        0 = circle, 1 = figure-eight, 2 = lawnmower
 */
void depth_nav_set_trajectory(uint8_t traj_id)
{
  switch (traj_id) {
    case 0: active_trajectory = TRAJ_CIRCLE;       break;
    case 1: active_trajectory = TRAJ_FIGURE_EIGHT; break;
    case 2: active_trajectory = TRAJ_LAWNMOWER;    break;
    default: active_trajectory = TRAJ_FIGURE_EIGHT; break;
  }
  // Reset trajectory progress when switching
  traj_angle    = 0.0f;
  lawn_y_offset = -traj_eight_scale;
  lawn_direction = 1;
  VERBOSE_PRINT("Trajectory set to: %d\n", active_trajectory);
}