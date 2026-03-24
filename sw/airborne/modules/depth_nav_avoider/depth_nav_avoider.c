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
 *
 * DEPTH SCALE NOTE:
 *  Model output:  raw float in [0, 255]
 *  Sender packs:  int16 = raw * 100  → wire values [0, 25500]
 *  Receiver unpacks: float = int16 / 100  → back to [0.0, 255.0]
 *  => depth_left/straight/right are in [0.0, 255.0] here.
 *  => Higher value = more free space (further obstacle).
 *  All thresholds below are expressed on this [0.0, 255.0] scale.
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

/*
 * Depth thresholds on the unpacked scale [0.0, 2.55].
 * Raw model output 0-255 → packed *100 as int16 → unpacked /100 as float.
 *
 *  safe_distance_threshold:    below this → hard stop + rotate
 *  caution_distance_threshold: below this → slow + gentle turn
 *
 * Scale is [0, 255]. Increase to react earlier; decrease for more aggressive flying.
 */
float safe_distance_threshold    = 130.0f;   // ~20% of range — hard obstacle, stop now
float caution_distance_threshold = 150.0f;  // ~40% of range — slow down, start turning

// Depth noise filter (0 = no filtering, 1 = never updates)
// 0.4 means 40% new reading, 60% history
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

// How close drone must be to trajectory waypoint before "on path" [m]
#define TRAJ_SNAP_DISTANCE 0.5f

/* ================================ */
/*    Global State Variables        */
/* ================================ */

enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
enum trajectory_type_t  active_trajectory = TRAJ_FIGURE_EIGHT;

int16_t obstacle_free_confidence = 0;
float   heading_increment        = 15.0f;

// Depth readings from ABI — on [0.0, 2.55] scale, higher = more free space
float depth_left             = 0.0f;
float depth_straight         = 0.0f;
float depth_right            = 0.0f;
static float depth_straight_filtered = 0.0f;

// Trajectory state
static float traj_angle      = 0.0f;   // current angle along circle/eight [rad]
static int   lawn_direction  = 1;      // +1 or -1 for lawnmower sweep direction
static float lawn_y_offset   = 0.0f;   // current lawnmower lateral position [m]

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
 *
 * cv_estimate_depth sends: int16 = raw_float * 100
 * So we unpack as:         float = int16 / 100.0f
 *
 * Result is in [0.0, 2.55] — higher means obstacle is farther away (more free space).
 *
 * Mapping:  pixel_x      → depth_left
 *           pixel_y      → depth_straight
 *           pixel_width  → depth_right
 */
static void depth_detection_cb(uint8_t __attribute__((unused)) sender_id,
                                int16_t pixel_x,
                                int16_t pixel_y,
                                int16_t pixel_width,
                                int16_t __attribute__((unused)) pixel_height,
                                int32_t __attribute__((unused)) quality,
                                int16_t __attribute__((unused)) extra)
{
  depth_left     = (float)pixel_x     / 100.0f;
  depth_straight = (float)pixel_y     / 100.0f;
  depth_right    = (float)pixel_width / 100.0f;

  VERBOSE_PRINT("ABI recv → L:%.2f S:%.2f R:%.2f\n",
                depth_left, depth_straight, depth_right);
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
  traj_angle    = 0.0f;
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
  // Higher alpha = more responsive; lower = smoother but slower to react
  depth_straight_filtered = DEPTH_FILTER_ALPHA * depth_straight
                          + (1.0f - DEPTH_FILTER_ALPHA) * depth_straight_filtered;

  VERBOSE_PRINT("Depths L:%.2f S:%.2f(f:%.2f) R:%.2f | Conf:%d | State:%d | Traj:%d\n",
                depth_left, depth_straight, depth_straight_filtered, depth_right,
                obstacle_free_confidence, navigation_state, active_trajectory);

  float moveDist = maxDistance;

  // ---- Update confidence based on filtered forward depth ----
  // depth_straight_filtered is in [0.0, 2.55]: higher = more free space
  if (depth_straight_filtered >= caution_distance_threshold) {
    // Path looks clear → gain confidence
    obstacle_free_confidence++;
  } else {
    // Something ahead → lose confidence faster than we gain it
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
        VERBOSE_PRINT("Out of bounds!\n");
        break;
      }

      // Check for approaching obstacle
      if (depth_straight_filtered < caution_distance_threshold) {
        navigation_state = CAUTION;
        trajectory_active = false;
        VERBOSE_PRINT("Entering CAUTION, filtered depth: %.2f\n",
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
      // Slow down and apply a gentle corrective turn toward the clearer side
      chooseDirectionalIncrement();
      increase_nav_heading(heading_increment * 0.5f);   // gentle — half the normal step
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist * 0.5f);
      moveWaypointForward(WP_GOAL,       moveDist * 0.5f);
      moveWaypointForward(WP_RETREAT,   -moveDist * 0.5f);

      // Obstacle too close → escalate to OBSTACLE_FOUND
      if (depth_straight_filtered < safe_distance_threshold) {
        // Pick the side with more free space (higher depth = more space)
        if (depth_left > depth_right) {
          heading_increment = -15.0f;  // CCW → turn left
          VERBOSE_PRINT("More space LEFT (L:%.2f > R:%.2f), turning CCW\n",
                        depth_left, depth_right);
        } else {
          heading_increment =  15.0f;  // CW → turn right
          VERBOSE_PRINT("More space RIGHT (R:%.2f >= L:%.2f), turning CW\n",
                        depth_right, depth_left);
        }
        obstacle_free_confidence = 0;
        navigation_state = OBSTACLE_FOUND;
        VERBOSE_PRINT("Escalating to OBSTACLE_FOUND, depth: %.2f\n",
                      depth_straight_filtered);
        break;
      }

      // Path has cleared → back to SAFE
      if (depth_straight_filtered >= caution_distance_threshold &&
          obstacle_free_confidence >= 2) {
        navigation_state = SAFE;
        VERBOSE_PRINT("CAUTION resolved, returning to SAFE\n");
      }
      break;
    }

    case OBSTACLE_FOUND: {
      // Freeze forward motion — hold position
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      // Back away from the obstacle
      moveWaypointForward(WP_RETREAT, -moveDist * 0.25f);

      // Rotate toward the clearer side, re-evaluated every cycle
      /*
      if (depth_left > depth_right) {
        heading_increment = -5.0f;  // CCW → turn left
      } else {
        heading_increment =  5.0f;  // CW → turn right
      }
      */
      increase_nav_heading(-heading_increment);

      // Once the path ahead is safe enough, begin searching for a clear heading
      if (depth_straight_filtered >= safe_distance_threshold &&
          obstacle_free_confidence >= 1) {
        navigation_state = SEARCH_FOR_SAFE_HEADING;
        VERBOSE_PRINT("Obstacle cleared, searching for safe heading\n");
      }
      break;
    }

    case SEARCH_FOR_SAFE_HEADING: {
      // Keep turning toward the clearer side while re-evaluating each cycle
      increase_nav_heading(heading_increment);

      // Only resume normal flight once we have good confidence the path is clear
      if (depth_straight_filtered >= caution_distance_threshold &&
          obstacle_free_confidence >= 2) {
        VERBOSE_PRINT("Safe heading found (depth:%.2f conf:%d), resuming SAFE\n",
                      depth_straight_filtered, obstacle_free_confidence);
        navigation_state = SAFE;

        // Snap trajectory angle to nearest point so we rejoin smoothly
        float cur_x = stateGetPositionEnu_f()->x;
        float cur_y = stateGetPositionEnu_f()->y;
        if (active_trajectory == TRAJ_CIRCLE || active_trajectory == TRAJ_FIGURE_EIGHT) {
          // 1. Calculate the angle to our current position
          traj_angle = atan2f(cur_y - arena_centre_y, cur_x - arena_centre_x);
          
          // 2. THE FIX: Add a "Boost" to skip past the obstacle!
          // 0.8 radians is roughly 45 degrees. This forces the target waypoint 
          // to jump forward along the curve, safely past whatever we just dodged.
          traj_angle += 0.8f; 
        } 
        else if (active_trajectory == TRAJ_LAWNMOWER) {
          // THE FIX FOR LAWNMOWER: 
          // If an obstacle blocks our current sweep, trying to push through it is dangerous.
          // Instead, we abandon this row and immediately step laterally to the next pass.
          lawn_direction *= -1;
          lawn_y_offset  += traj_lawn_step * lawn_direction;
          
          // Clamp to arena bounds just like in the normal update function
          if (lawn_y_offset > traj_eight_scale)  lawn_y_offset =  traj_eight_scale;
          if (lawn_y_offset < -traj_eight_scale) lawn_y_offset = -traj_eight_scale;
        }
      }
      break;
    }

    case OUT_OF_BOUNDS: {
      // Minimum cycles in OOB before checking re-entry — prevents thrashing
      static uint8_t oob_cycles = 0;
      oob_cycles++;

      // Turn back toward centre (always CW for simplicity, can be made directional)
      increase_nav_heading(15.0f * (heading_increment >= 0 ? 1.0f : -1.0f));
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist);
      moveWaypointForward(WP_GOAL,       moveDist * 0.5f);
      moveWaypointForward(WP_RETREAT,   -moveDist * 0.5f);

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
      float cur_x = stateGetPositionEnu_f()->x;
      float cur_y = stateGetPositionEnu_f()->y;

      float sweep_end_x = arena_centre_x + lawn_direction * traj_eight_scale;
      float sweep_end_y = arena_centre_y + lawn_y_offset;

      float dx = sweep_end_x - cur_x;
      float dy = sweep_end_y - cur_y;
      float dist_to_end = sqrtf(dx*dx + dy*dy);

      if (dist_to_end < TRAJ_SNAP_DISTANCE) {
        // Reached end of pass — reverse direction and step laterally
        lawn_direction *= -1;
        lawn_y_offset  += traj_lawn_step * (float)lawn_direction;

        // Clamp to arena bounds
        if (lawn_y_offset >  traj_eight_scale) lawn_y_offset =  traj_eight_scale;
        if (lawn_y_offset < -traj_eight_scale) lawn_y_offset = -traj_eight_scale;

        VERBOSE_PRINT("LAWNMOWER: reversing, new y_offset: %.2f\n", lawn_y_offset);
      }

      target_x = sweep_end_x;
      target_y = sweep_end_y;

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
 * @brief Adjusts navigation heading by a given increment in degrees.
 *        Positive = CW (right), Negative = CCW (left) in NED heading convention.
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
 * @brief Uses current depth readings to pick the turn direction with more space.
 *
 * Depth values are in [0.0, 2.55] — directly comparable, no fabsf needed.
 * Higher depth → more free space on that side.
 * Positive heading_increment = CW (right), negative = CCW (left).
 */
static void chooseDirectionalIncrement(void)
{
  if (depth_left > depth_right) {
    heading_increment = -5.0f;   // CCW — more free space to the left
    VERBOSE_PRINT("chooseDir: LEFT clearer (L:%.2f > R:%.2f), increment = -5\n",
                  depth_left, depth_right);
  } else if (depth_right > depth_left) {
    heading_increment =  5.0f;   // CW  — more free space to the right
    VERBOSE_PRINT("chooseDir: RIGHT clearer (R:%.2f > L:%.2f), increment = +5\n",
                  depth_right, depth_left);
  } else {
    // Exactly equal or both zero — pick randomly to break symmetry
    heading_increment = (rand() % 2 == 0) ? 5.0f : -5.0f;
    VERBOSE_PRINT("chooseDir: equal depths, random increment: %.1f\n",
                  heading_increment);
  }
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
  traj_angle     = 0.0f;
  lawn_y_offset  = -traj_eight_scale;
  lawn_direction = 1;
  VERBOSE_PRINT("Trajectory set to: %d\n", active_trajectory);
}