/*
 * Depth Nav Avoider — Random Walk Edition
 *
 * Logic is intentionally simple:
 *  - Fly forward on current heading
 *  - If something ahead → turn toward clearer side
 *  - If out of bounds   → turn toward arena centre
 *  - No trajectory, no fixed path — purely reactive
 *
 * DEPTH SCALE:
 *  Model output [0, 255] → packed *100 as int16 → unpacked /100 as float
 *  => depth values here are [0.0, 255.0], higher = more free space
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
  SAFE,               // flying forward freely
  OBSTACLE_FOUND,     // obstacle ahead — rotating away
  OUT_OF_BOUNDS       // outside geofence — turning back
};

/* ================================ */
/*    Median Filter                 */
/* ================================ */

#define MEDIAN_WINDOW 7

static float buf_left[MEDIAN_WINDOW];
static float buf_straight[MEDIAN_WINDOW];
static float buf_right[MEDIAN_WINDOW];
static uint8_t median_idx = 0;

static float median_of(const float *buf)
{
  float tmp[MEDIAN_WINDOW];
  memcpy(tmp, buf, MEDIAN_WINDOW * sizeof(float));
  for (int i = 1; i < MEDIAN_WINDOW; i++) {
    float key = tmp[i];
    int j = i - 1;
    while (j >= 0 && tmp[j] > key) { tmp[j+1] = tmp[j]; j--; }
    tmp[j+1] = key;
  }
  return tmp[MEDIAN_WINDOW / 2];
}

/* ================================ */
/*    Tunable Parameters            */
/* ================================ */

/*
 * Based on observed data: clear ~80-180, near wall ~10-40.
 * safe:   below this = hard stop and rotate immediately
 * caution: below this = start turning
 * Tune these if the drone is still too aggressive or too timid.
 */
float safe_distance_threshold    = 40.0f;
float caution_distance_threshold = 55.0f;

float maxDistance = 0.5f;  // waypoint step size [m]

/*
 * After avoiding, fly straight for at least this many cycles
 * before being allowed to pick a new random heading.
 * Prevents the drone from spinning on the spot.
 */
#define MIN_SAFE_CYCLES 10

/*
 * Random heading change range [degrees].
 * Each avoidance adds a random extra offset on top of turning toward
 * the clearer side, so the drone doesn't get stuck in loops.
 */
float TURN_BASE_DEG   = 8.0f;   // base rotation per cycle when avoiding
float RANDOM_EXTRA_DEG = 12.0f;  // random extra heading offset after clearing

/* ================================ */
/*    State Variables               */
/* ================================ */

enum navigation_state_t navigation_state = SAFE;

// Raw depth from ABI
float depth_left     = 0.0f;
float depth_straight = 0.0f;
float depth_right    = 0.0f;

// Median-smoothed depth — used for all decisions
static float smooth_left     = 0.0f;
static float smooth_straight = 0.0f;
static float smooth_right    = 0.0f;

// Counts cycles spent in SAFE to enforce MIN_SAFE_CYCLES
static uint16_t safe_cycle_count = 0;

// Arena centre for out-of-bounds recovery
static float arena_centre_x = 0.0f;
static float arena_centre_y = 0.0f;

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

  // Update median on every camera frame for fast response
  buf_left[median_idx]     = depth_left;
  buf_straight[median_idx] = depth_straight;
  buf_right[median_idx]    = depth_right;
  median_idx = (median_idx + 1) % MEDIAN_WINDOW;

  smooth_left     = median_of(buf_left);
  smooth_straight = median_of(buf_straight);
  smooth_right    = median_of(buf_right);

  VERBOSE_PRINT("ABI → Raw L:%.1f S:%.1f R:%.1f | Smooth L:%.1f S:%.1f R:%.1f\n",
                depth_left, depth_straight, depth_right,
                smooth_left, smooth_straight, smooth_right);
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

  // Pre-fill buffers above caution threshold — no false trigger on startup
  for (int i = 0; i < MEDIAN_WINDOW; i++) {
    buf_left[i]     = caution_distance_threshold + 20.0f;
    buf_straight[i] = caution_distance_threshold + 20.0f;
    buf_right[i]    = caution_distance_threshold + 20.0f;
  }
  smooth_left = smooth_straight = smooth_right = caution_distance_threshold + 20.0f;
  median_idx = 0;

  arena_centre_x = WaypointX(WP_CENTER);
  arena_centre_y = WaypointY(WP_CENTER);

  navigation_state = SAFE;
  safe_cycle_count = 0;

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

  // Always-on print for tuning — remove when happy
  fprintf(stderr, "[DNA] Smooth L:%.1f S:%.1f R:%.1f | State:%d\n",
          smooth_left, smooth_straight, smooth_right, (int)navigation_state);

  switch (navigation_state) {

    case SAFE: {
      // Probe waypoint ahead to check geofence
      moveWaypointForward(WP_TRAJECTORY, maxDistance);

      // Out of bounds — redirect immediately
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),
                              WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
        VERBOSE_PRINT("Out of bounds, turning to centre\n");
        break;
      }

      // Obstacle ahead — start rotating away
      if (smooth_straight < caution_distance_threshold) {
        navigation_state = OBSTACLE_FOUND;
        VERBOSE_PRINT("Obstacle! smooth_S=%.1f, turning toward clearer side\n",
                      smooth_straight);
        break;
      }

      // All clear — fly forward
      moveWaypointForward(WP_GOAL, maxDistance);
      safe_cycle_count++;

      // After enough safe cycles, nudge heading by a small random amount
      // so the drone doesn't just drill a straight line into the same pole
      if (safe_cycle_count >= MIN_SAFE_CYCLES) {
        float nudge = ((float)(rand() % 21) - 10.0f);  // -10 to +10 degrees
        increase_nav_heading(nudge);
        safe_cycle_count = 0;
        VERBOSE_PRINT("Random nudge: %.1f deg\n", nudge);
      }
      break;
    }

    case OBSTACLE_FOUND: {
      // Freeze position
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      // Turn toward whichever side has more free space
      float turn_dir = (smooth_left > smooth_right) ? -1.0f : 1.0f;
      increase_nav_heading(turn_dir * TURN_BASE_DEG);

      VERBOSE_PRINT("Rotating %.0f deg/cycle, L:%.1f R:%.1f\n",
                    turn_dir * TURN_BASE_DEG, smooth_left, smooth_right);

      // Once path is clear, add a random extra offset and resume
      if (smooth_straight >= caution_distance_threshold) {
        // Random extra turn so we don't immediately re-approach the same obstacle
        float extra = turn_dir * (RANDOM_EXTRA_DEG + (float)(rand() % 31));
        increase_nav_heading(extra);
        VERBOSE_PRINT("Path clear, extra turn %.1f deg, back to SAFE\n", extra);

        navigation_state = SAFE;
        safe_cycle_count = 0;
      }
      break;
    }

    case OUT_OF_BOUNDS: {
      // Point toward arena centre and hold position
      waypoint_move_here_2d(WP_GOAL);
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

void depth_nav_set_trajectory(uint8_t traj_id __attribute__((unused)))
{
  // Trajectories removed — random walk only
  VERBOSE_PRINT("depth_nav_set_trajectory called but ignored (random walk mode)\n");
}