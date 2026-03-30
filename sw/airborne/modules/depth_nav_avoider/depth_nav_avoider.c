/*
 * Depth Nav Avoider Module
 * Merges the confidence-based state machine from orange_avoider
 * with the depth-aware, directional turning from depth_avoider.
 * Adds trajectory following: circle, figure-eight, and lawnmower.
 *
 * Key features:
 *  - Uses depth (left/straight/right) for smarter turn decisions
 *  - All three depth channels are low-pass filtered
 *  - Confidence system to avoid false positives
 *  - CAUTION state for gradual obstacle response
 *  - RETREAT waypoint for backing away from obstacles
 *  - Trajectory following with return-to-path after avoidance
 *  - OUT_OF_BOUNDS minimum cycle counter to prevent thrashing
 *  - Obstacle-arc skipping to avoid re-entering obstacle zone
 *  - Direction deadband prevents flip-flopping on noise
 *
 * DEPTH SCALE NOTE:
 *  Model output:  raw float in [0, 255]
 *  Sender packs:  int16 = raw * 100  -> wire values [0, 25500]
 *  Receiver unpacks: float = int16 / 100  -> back to [0.0, 255.0]
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
#include <string.h>
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
/*    State Machine          */
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
 * Depth thresholds on the unpacked scale [0.0, 255.0].
 *
 *  safe_distance_threshold:    below this -> hard stop + rotate
 *  caution_distance_threshold: below this -> slow + gentle turn
 *
 * Scale is [0, 255]. Increase to react earlier; decrease for more aggressive flying.
 */
float safe_distance_threshold    = 55.0f;
float caution_distance_threshold = 60.0f;

/* Low-pass filter alpha for all three depth channels.
 * 0 = frozen (no update), 1 = raw pass-through (no smoothing).
 * Tunable live from GCS. */
float depth_filter_alpha = 0.6f;

/* Max forward displacement per cycle [m] */
float maxDistance = 0.5f;

/* Confidence system */
const int16_t max_trajectory_confidence = 3;

/* Trajectory parameters — all tunable live from GCS */
float traj_circle_radius  = 1.8f;
float traj_circle_speed   = 0.05f;
float traj_eight_scale    = 1.6f;
float traj_lawn_step      = 0.5f;

#define TRAJ_SNAP_DISTANCE 0.5f

/* ================================ */
/*    State Variables               */
/* ================================ */

enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
enum trajectory_type_t  active_trajectory = TRAJ_FIGURE_EIGHT;

int16_t obstacle_free_confidence = 0;
float   heading_increment        = 15.0f;

/* Raw depth readings from ABI */
float depth_left     = 0.0f;
float depth_straight = 0.0f;
float depth_right    = 0.0f;

/* Filtered depth readings — all three channels */
static float depth_left_filtered     = 0.0f;
static float depth_straight_filtered = 0.0f;
static float depth_right_filtered    = 0.0f;

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
/*    ABI                           */
/* ================================ */

#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif
static abi_event depth_detection_ev;

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
}

/* ================================ */
/*    Helpers                       */
/* ================================ */

static void moveWaypointForward(uint8_t waypoint, float distanceMeters);
static void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static void moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static void increase_nav_heading(float incrementDegrees);
static void set_nav_heading_towards(float tx, float ty);

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

/* ================================ */
/*    Init                          */
/* ================================ */

void depth_nav_avoider_init(void)
{
  srand(time(NULL));

  /* Initialise all filters at threshold so first cycles don't false-trigger */
  depth_left_filtered     = caution_distance_threshold;
  depth_straight_filtered = caution_distance_threshold;
  depth_right_filtered    = caution_distance_threshold;

  arena_centre_x = WaypointX(WP_CENTER);
  arena_centre_y = WaypointY(WP_CENTER);

  traj_angle     = 0.0f;
  obstacle_angle = 0.0f;
  lawn_y_offset  = -traj_eight_scale;

  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID,
                             &depth_detection_ev,
                             depth_detection_cb);

  VERBOSE_PRINT("Init done. Arena centre (%.2f, %.2f)\n",
                arena_centre_x, arena_centre_y);
}

/* ================================ */
/*    Periodic                      */
/* ================================ */

void depth_nav_avoider_periodic(void)
{
  if (!autopilot_in_flight()) { return; }

  /* ---- Low-pass filter all three depth channels ---- */
  depth_left_filtered     = depth_filter_alpha * depth_left
                          + (1.0f - depth_filter_alpha) * depth_left_filtered;
  depth_straight_filtered = depth_filter_alpha * depth_straight
                          + (1.0f - depth_filter_alpha) * depth_straight_filtered;
  depth_right_filtered    = depth_filter_alpha * depth_right
                          + (1.0f - depth_filter_alpha) * depth_right_filtered;

  VERBOSE_PRINT("Depths L:%.2f(f:%.2f) S:%.2f(f:%.2f) R:%.2f(f:%.2f) | Conf:%d | State:%d | Traj:%d\n",
                depth_left, depth_left_filtered,
                depth_straight, depth_straight_filtered,
                depth_right, depth_right_filtered,
                obstacle_free_confidence, navigation_state, active_trajectory);

  float moveDist = maxDistance;

  /* ---- Update confidence based on filtered forward depth ---- */
  if (depth_straight_filtered >= caution_distance_threshold) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  /* ---- Emergency override: raw depth dangerously low while in SAFE ---- */
  if (navigation_state == SAFE && depth_straight < safe_distance_threshold) {
    VERBOSE_PRINT("EMERGENCY: raw depth %.2f below safe threshold\n", depth_straight);

    obstacle_angle = traj_angle;

    if (depth_left_filtered > depth_right_filtered) {
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
        VERBOSE_PRINT("Out of bounds, turning to centre\n");
        break;
      }

      if (depth_straight_filtered < caution_distance_threshold) {
        chooseDirectionalIncrement();  /* pick direction ONCE on entry */
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
      /* Direction locked on entry — don't re-evaluate every cycle */
      increase_nav_heading(heading_increment * 0.5f);
      moveWaypointForward(WP_TRAJECTORY, 0.35f * moveDist * 0.5f);
      moveWaypointForward(WP_GOAL,       moveDist * 0.5f);
      moveWaypointForward(WP_RETREAT,   -moveDist * 0.5f);

      /* Obstacle too close -> escalate to OBSTACLE_FOUND */
      if (depth_straight_filtered < safe_distance_threshold) {
        obstacle_angle = traj_angle;

        if (depth_left_filtered > depth_right_filtered) {
          heading_increment = -15.0f;
          VERBOSE_PRINT("More space LEFT (L:%.2f > R:%.2f), turning CCW\n",
                        depth_left_filtered, depth_right_filtered);
        } else {
          heading_increment =  15.0f;
          VERBOSE_PRINT("More space RIGHT (R:%.2f >= L:%.2f), turning CW\n",
                        depth_right_filtered, depth_left_filtered);
        }
        obstacle_free_confidence = 0;
        navigation_state = OBSTACLE_FOUND;
        VERBOSE_PRINT("Obstacle! smooth_S=%.1f, turning toward clearer side\n",
                      smooth_straight);
        break;
      }

      /* Path cleared -> back to SAFE */
      if (depth_straight_filtered >= caution_distance_threshold &&
          obstacle_free_confidence >= 2) {
        navigation_state = SAFE;
        VERBOSE_PRINT("CAUTION resolved, returning to SAFE\n");
      }
      break;
    }

    case OBSTACLE_FOUND: {
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);
      moveWaypointForward(WP_RETREAT, -moveDist * 0.25f);

      increase_nav_heading(heading_increment);

      if (depth_straight_filtered >= safe_distance_threshold &&
          obstacle_free_confidence >= 2) {
        navigation_state = SEARCH_FOR_SAFE_HEADING;
        VERBOSE_PRINT("Obstacle cleared, searching for safe heading\n");
      }
      break;
    }

    case SEARCH_FOR_SAFE_HEADING: {
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);
      increase_nav_heading(heading_increment);

      if (depth_straight_filtered >= caution_distance_threshold &&
          obstacle_free_confidence >= 3) {
        VERBOSE_PRINT("Safe heading found (depth:%.2f conf:%d), resuming SAFE\n",
                      depth_straight_filtered, obstacle_free_confidence);
        navigation_state = SAFE;

        float cur_x = stateGetPositionEnu_f()->x;
        float cur_y = stateGetPositionEnu_f()->y;

        if (active_trajectory == TRAJ_CIRCLE || active_trajectory == TRAJ_FIGURE_EIGHT) {
          float cur_angle  = atan2f(cur_y - arena_centre_y, cur_x - arena_centre_x);
          float skip_angle = obstacle_angle + 2.0f;
          FLOAT_ANGLE_NORMALIZE(skip_angle);

          float dist_cur  = cur_angle  - obstacle_angle;
          float dist_skip = skip_angle - obstacle_angle;
          if (dist_cur  < 0) dist_cur  += 2.0f * M_PI;
          if (dist_skip < 0) dist_skip += 2.0f * M_PI;

          if (dist_cur > dist_skip) {
            traj_angle = cur_angle + 0.5f;
          } else {
            traj_angle = skip_angle;
          }
          FLOAT_ANGLE_NORMALIZE(traj_angle);

          VERBOSE_PRINT("Rejoin: obstacle=%.2f skip=%.2f traj=%.2f\n",
                        obstacle_angle, skip_angle, traj_angle);
        }
        else if (active_trajectory == TRAJ_LAWNMOWER) {
          lawn_y_offset += traj_lawn_step * (float)lawn_direction;
          lawn_direction *= -1;

          if (lawn_y_offset >  traj_eight_scale) lawn_y_offset =  traj_eight_scale;
          if (lawn_y_offset < -traj_eight_scale) lawn_y_offset = -traj_eight_scale;

          VERBOSE_PRINT("Lawnmower: skipped row, y_offset=%.2f dir=%d\n",
                        lawn_y_offset, lawn_direction);
        }
      }
      break;
    }

    case OUT_OF_BOUNDS: {
      static uint8_t oob_cycles = 0;
      oob_cycles++;

      set_nav_heading_towards(arena_centre_x, arena_centre_y);

      // Re-check with probed waypoint
      moveWaypointForward(WP_TRAJECTORY, maxDistance);
      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY),
                             WaypointY(WP_TRAJECTORY))) {
        VERBOSE_PRINT("Back inside zone, resuming SAFE\n");
        navigation_state = SAFE;
        safe_cycle_count = 0;
      }
      break;
    }

    default: break;
  }
}

/* ================================ */
/*    Stub — kept for flight plan   */
/*    compatibility                 */
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
        lawn_y_offset += traj_lawn_step * (float)lawn_direction;
        lawn_direction *= -1;

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

/**
 * @brief Pick turn direction based on filtered left vs right depth.
 *        Deadband of 5.0 (on [0,255] scale) prevents flip-flopping on noise.
 *        If difference is within deadband, keeps previous heading_increment.
 */
static void chooseDirectionalIncrement(void)
{
  float diff = depth_left_filtered - depth_right_filtered;
  if (diff > 5.0f) {
    heading_increment = -5.0f;
    VERBOSE_PRINT("chooseDir: LEFT clearer (Lf:%.2f > Rf:%.2f), increment = -5\n",
                  depth_left_filtered, depth_right_filtered);
  } else if (diff < -5.0f) {
    heading_increment =  5.0f;
    VERBOSE_PRINT("chooseDir: RIGHT clearer (Rf:%.2f > Lf:%.2f), increment = +5\n",
                  depth_right_filtered, depth_left_filtered);
  }
  /* else: keep previous heading_increment — don't flip-flop */
}

/* ================================ */
/*    Public Trajectory Selector    */
/* ================================ */

void depth_nav_set_trajectory(uint8_t traj_id)
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