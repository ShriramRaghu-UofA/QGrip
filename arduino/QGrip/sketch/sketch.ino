// MyoHandiHand - Position Controller (UNO Q / STM32 port)
//
// Adapted from: Handi-Hand Individual Digit Velocity Controller (Grasp_Vel_rev1)
// Original author: Michael (Rory) Dawson - https://github.com/BLINCdev/HANDi-Hand
//   https://github.com/BLINCdev/HANDi-Hand/blob/master/Software/Grasp_Vel_rev1/Grasp_Vel_rev1.ino
//
// Objective: Drive the Handi Hand's 6 digit joints to a raw goal position sent
// per-joint in a single packet by RPC from the Python app over the Arduino UNO Q
// Router Bridge. Each joint is independently wired to EITHER a DYNAMIXEL X-series
// servo (position mode, with a per-joint profile velocity) OR a hobby (analog RC)
// PWM servo - see the JOINT_INTERFACE table below. This lets a single sketch
// drive a hand that mixes both actuator types (e.g. DYNAMIXELs on the digits,
// hobby servos on the thumb, or any other combination) without maintaining two
// near-identical sketches. The sketch performs no range/limit checking of its
// own - it writes whatever position it's given straight to the servo. Grip
// shapes, "closed hand" presets, per-digit drive limits, and any other range
// gating live entirely on the RPC caller side (see rpc_handi.py) so they can be
// tuned without reflashing.
//
// Mix-and-match architecture:
// Support for each interface is compiled in independently based on whether any
// joint in JOINT_INTERFACE[] below uses it - DYNAMIXEL_ENABLED and
// HOBBY_SERVO_ENABLED are computed from that table, not hand-set. This keeps a
// DYNAMIXEL-only or hobby-only build from paying for (or even compiling) the
// other interface's library/bus/pins - set every joint to one interface to get
// back the old single-interface sketch's behavior exactly.
//
// The XL330-M288-T servos are pre-configured in the DYNAMIXEL Wizard 2.0 software to
// be in position mode, with their own CW/CCW joint limits set directly on the servo
// (via DYNAMIXEL Wizard 2.0's EEPROM Min/Max Position Limit, not by this sketch). An
// individual (collision-free) ping sweep of IDs 1-20 confirmed 6 servos on the bus at
// IDs 10-15 (see JOINT_INTERFACE[].dxl.id) - the joint_id 1-6 used over the
// Bridge/RPC is a logical index into JOINT_INTERFACE[], independent of the physical
// DYNAMIXEL ID.
//
// Definitions (6 digits, matching the original reference 1:1):
// joint 1: thumb rotate
// joint 2: thumb flex
// joint 3: index digit
// joint 4: middle digit
// joint 5: ring digit
// joint 6: baby digit
//
// DYNAMIXEL wiring (no DYNAMIXEL Shield, no TTL adapter - direct single-wire bus):
// DYNAMIXEL's data line is single-wire half-duplex. With no shield/adapter to do
// the TX/RX merging in hardware, both D21 (TX / PB10) and D20 (RX / PB11) - the
// pins behind the "Serial3" object on this board, the only free hardware UART
// (Serial1 is the boot console; Serial/Serial2 belong to the Router Bridge/Monitor
// link to the MPU) - are wired together to the single DATA pin on the DYNAMIXEL,
// with a small series resistor (~150-470 ohm) on the TX leg to avoid driver
// contention while RX is also actively driven by the bus. Connect a 5V/4A PSU to
// power the bus, and GND to GND.
//
// Because TX and RX share one wire, everything the MCU transmits is echoed straight
// back into its own RX ahead of the servo's real reply. See half_duplex_echo_serial.h
// for how that echo is discarded before the Dynamixel2Arduino library ever sees it.
// There's still no shield DIR pin to toggle, so the library itself is used in its
// full-duplex mode (DXL_DIR_PIN = -1) - the echo handling lives entirely in the
// HalfDuplexEchoSerial wrapper below, not in the library.
//
// Hobby (analog RC) servo wiring: any joint set to HOBBY in JOINT_INTERFACE[] is
// driven by a plain PWM signal wire on the Arduino header pin given by
// .hobby.pin, using this board's native Zephyr PWM channels (see
// bring_up_hobby_servo() below for the PWM caveat re: the devicetree's default
// period being overridden at runtime).
//
// RPC contract (MPU -> MCU, via Arduino_RouterBridge):
//   Bridge.call("set_positions", positions) -> bool
//     positions: array of 6 raw goal positions, one per joint_id 1-6 (see
//                Definitions above) - DYNAMIXEL units for a DYNAMIXEL joint, pulse
//                width in microseconds for a hobby-servo joint (see
//                JOINT_INTERFACE[] to check which is which). Sent to each enabled
//                joint as-is; a hobby-servo joint clamps to the servo-safe PWM
//                range (see SERVO_PULSE_MIN_US/MAX_US), a DYNAMIXEL joint gets no
//                range checking at all - drive-limit gating for both is the
//                caller's job (see rpc_handi.py). Non-blocking. Always returns
//                true.
//     A separate call from set_led_frame (below) - RPClite's packet framing
//     scanner re-walks nested-array arguments on every incoming byte with no
//     memoization, so a combined positions+led_frame payload never finished
//     framing within the client's timeout. Keep this one small.
//   Bridge.call("set_led_frame", led_frame) -> bool
//     led_frame: 104 grayscale values (0-7), one byte per pixel, row-major, sent
//                as a MessagePack bin blob (Python: bytes(...), not a list) - a
//                104-element plain array here reliably wedged the whole RPC link
//                on its second call. Enqueued for led_thread (see its definition
//                near `matrix` above), which draws it independently - this
//                handler never blocks on matrix.draw(). Always returns true.
//   Bridge.call("move_joint", joint_id, goal_position) -> bool
//     Manual single-joint move, e.g. for bench testing. joint_id: 1-6, see
//     Definitions above. goal_position sent as-is for a DYNAMIXEL joint (no
//     clamping), clamped to SERVO_PULSE_MIN_US/MAX_US for a hobby-servo joint.
//     Returns true if joint_id was valid and enabled.
//   Bridge.call("get_position", joint_id) -> [position, velocity]
//     DYNAMIXEL joint: read-only live servo read (not last commanded value).
//     Hobby-servo joint: hobby servos have no feedback line, so this returns the
//     last commanded pulse width instead (state this sketch tracks on the
//     caller's behalf, not a live read), and velocity is always 0. Returns
//     [-1, -1] if joint_id was invalid or not yet enabled.
//
// References:
// Structs: http://playground.arduino.cc/Code/Struct
// OpenRB-150: https://emanual.robotis.com/docs/en/parts/controller/openrb-150/
// Dynamixel2Arduino: https://github.com/ROBOTIS-GIT/Dynamixel2Arduino
// Arduino_RouterBridge: https://github.com/arduino-libraries/Arduino_RouterBridge

#include <array>

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

const int DIGIT_NUM = 6;
const int LED_MATRIX_PIXELS = 104;

// ---------------------------------------------------------------------------
// Per-joint interface selection. This is the single source of truth for the
// mix-and-match architecture: each joint_id (1-6, see Definitions above) picks
// exactly one interface via the JOINT<n>_IS_DYNAMIXEL #defines below. These have
// to be preprocessor #defines (not a runtime table) so that DYNAMIXEL_ENABLED /
// HOBBY_SERVO_ENABLED can be derived from them with the preprocessor - a build
// with every joint on one interface then compiles out the other interface's
// library/bus/pins entirely (its #include, globals, and RPC-handler branches
// never exist in the translation unit).
//
// To reassign a joint, flip its JOINT<n>_IS_DYNAMIXEL 1/0 below and set the
// matching id/pin in JOINT_INTERFACE[] further down.
// ---------------------------------------------------------------------------
#define JOINT1_IS_DYNAMIXEL 1  // thumb rotate
#define JOINT2_IS_DYNAMIXEL 1  // thumb flex
#define JOINT3_IS_DYNAMIXEL 1  // index digit
#define JOINT4_IS_DYNAMIXEL 1  // middle digit
#define JOINT5_IS_DYNAMIXEL 1  // ring digit
#define JOINT6_IS_DYNAMIXEL 1  // baby digit

// DYNAMIXEL_ENABLED / HOBBY_SERVO_ENABLED gate every bit of code specific to
// that interface (library includes, bus init, RPC-handler branches) so a sketch
// with no joint on an interface never links or initializes that interface's bus
// at all.
#if JOINT1_IS_DYNAMIXEL || JOINT2_IS_DYNAMIXEL || JOINT3_IS_DYNAMIXEL || \
    JOINT4_IS_DYNAMIXEL || JOINT5_IS_DYNAMIXEL || JOINT6_IS_DYNAMIXEL
#define DYNAMIXEL_ENABLED 1
#endif

#if !JOINT1_IS_DYNAMIXEL || !JOINT2_IS_DYNAMIXEL || !JOINT3_IS_DYNAMIXEL || \
    !JOINT4_IS_DYNAMIXEL || !JOINT5_IS_DYNAMIXEL || !JOINT6_IS_DYNAMIXEL
#define HOBBY_SERVO_ENABLED 1
#endif

enum JointInterface { INTERFACE_DYNAMIXEL, INTERFACE_HOBBY_SERVO };

#define JOINT_IFACE(n) (JOINT##n##_IS_DYNAMIXEL ? INTERFACE_DYNAMIXEL : INTERFACE_HOBBY_SERVO)

struct JointConfig {
  JointInterface interface;
  int dxl_id;        // physical DYNAMIXEL ID on the bus - used when interface == INTERFACE_DYNAMIXEL
  pin_size_t hobby_pin;  // Arduino header pin driving the PWM signal line - used when interface == INTERFACE_HOBBY_SERVO
};

// Runtime mirror of the JOINT<n>_IS_DYNAMIXEL #defines above, plus each joint's
// physical id/pin. joint_id 1-6 indexes 1-based below (index 0 unused, kept for
// parity with the 1-based joint_id used throughout the RPC contract). dxl_id is
// only meaningful when that joint is DYNAMIXEL; hobby_pin only when it's a hobby
// servo - the unused field per row is left at 0.
const JointConfig JOINT_INTERFACE[DIGIT_NUM + 1] = {
  /* joint 0, unused */ { INTERFACE_DYNAMIXEL, 0, 0 },
  /* joint 1: thumb rotate */ { JOINT_IFACE(1), 10, 2 },
  /* joint 2: thumb flex   */ { JOINT_IFACE(2), 11, 3 },
  /* joint 3: index digit  */ { JOINT_IFACE(3), 12, 5 },
  /* joint 4: middle digit */ { JOINT_IFACE(4), 13, 6 },
  /* joint 5: ring digit   */ { JOINT_IFACE(5), 14, 8 },
  /* joint 6: baby digit   */ { JOINT_IFACE(6), 15, 9 },
};

#ifdef DYNAMIXEL_ENABLED
#include <Dynamixel2Arduino.h>
#include "half_duplex_echo_serial.h"

#define DXL_DIR_PIN  -1        // no shield DIR pin on UNO Q -> full duplex (library side)

const float DXL_PROTOCOL_VERSION = 2.0;

// D20(RX)/D21(TX) tied together onto DYNAMIXEL's single-wire DATA line; see
// half_duplex_echo_serial.h for why this wrapper is needed instead of using
// Serial3 directly.
HalfDuplexEchoSerial dxl_serial(Serial3);
Dynamixel2Arduino dxl(dxl_serial, DXL_DIR_PIN);

// Every dxl.* call goes through this bus, and it's touched from two different
// threads: RPC handlers (move_joint/set_positions/get_position) run on the
// Bridge's own update thread (see Arduino_RouterBridge's bridge.h), while the
// missing-motor retry lives in loop() on the main sketch thread. The DYNAMIXEL
// link is single-wire half-duplex (see the wiring note above) - two threads
// writing/reading it at once would corrupt each other's transaction. Every dxl.*
// call site below takes this mutex first.
struct k_mutex dxl_bus_mutex;

// This namespace is required to use Control table item names
using namespace ControlTableItem;
#endif  // DYNAMIXEL_ENABLED

#ifdef HOBBY_SERVO_ENABLED
#include <zephyr/drivers/pwm.h>
#include <zephyrPinctrl.h>

// init_dev_apply_channel_pinctrl() / state_pin_index_from_spec_index() below live
// in this namespace (see zephyrPinctrl.h) - same helpers Arduino.h's own
// analogWrite() uses internally (cores/arduino/wiring_analog.cpp).
using namespace zephyr::arduino;

// Servo-safe pulse width bounds, in microseconds. 1000-2000us is the common
// analog-hobby-servo range (500-2500us "extended" range servos exist but aren't
// assumed here).
const uint32_t SERVO_PULSE_MIN_US = 800;
const uint32_t SERVO_PULSE_MAX_US = 1900;
const uint32_t SERVO_PULSE_CENTER_US = 1500;

// PWM frame period this sketch drives every hobby-servo channel at, overriding
// the devicetree's PWM_HZ(500) default per-call via pwm_set_dt() - every
// Arduino-header PWM channel on this board is declared at PWM_HZ(500) in the
// overlay (2000000ns/2ms period, 500Hz refresh), not left at that default here.
// 20ms = 50Hz, standard analog-hobby-servo refresh rate.
const uint32_t SERVO_PWM_PERIOD_US = 20000;

uint32_t clamp_pulse_us(int32_t us) {
  if (us < (int32_t)SERVO_PULSE_MIN_US) return SERVO_PULSE_MIN_US;
  if (us > (int32_t)SERVO_PULSE_MAX_US) return SERVO_PULSE_MAX_US;
  return (uint32_t)us;
}

// One pwm_dt_spec per Arduino-header PWM channel declared in this board's
// zephyr_user "pwms" devicetree property (see arduino_uno_q_stm32u585xx.overlay)
// - mirrors what Arduino.h's own analogWrite() builds internally
// (cores/arduino/wiring_analog.cpp), just re-declared here so this sketch can
// call pwm_set_pulse_dt() directly with a raw microsecond pulse width instead of
// going through analogWrite()'s 0-255 duty-cycle mapping, which isn't the right
// abstraction for a hobby servo's absolute pulse-width contract.
#define PWM_DT_SPEC(n, p, i) PWM_DT_SPEC_GET_BY_IDX(n, i),
const struct pwm_dt_spec pwm_specs[] = {
  DT_FOREACH_PROP_ELEM(DT_PATH(zephyr_user), pwms, PWM_DT_SPEC)
};
const size_t PWM_SPEC_COUNT = sizeof(pwm_specs) / sizeof(pwm_specs[0]);

// Find pwm_specs[]'s index for a given Arduino header pin number, matching this
// board's pwm-pin-gpios devicetree list (parallel array to pwms, same order).
#define PWM_PIN_GPIO(n, p, i)                                                              \
  DIGITAL_PIN_GPIOS_FIND_PIN(DT_REG_ADDR(DT_PHANDLE_BY_IDX(DT_PATH(zephyr_user), p, i)),   \
                              DT_PHA_BY_IDX(DT_PATH(zephyr_user), p, i, pin)),
const pin_size_t pwm_pin_gpios[] = {
  DT_FOREACH_PROP_ELEM(DT_PATH(zephyr_user), pwm_pin_gpios, PWM_PIN_GPIO)
};

size_t pwm_index_for_pin(pin_size_t pin) {
  size_t n = sizeof(pwm_pin_gpios) / sizeof(pwm_pin_gpios[0]);
  for (size_t i = 0; i < n && i < PWM_SPEC_COUNT; i++) {
    if (pwm_pin_gpios[i] == pin) {
      return i;
    }
  }
  return (size_t)-1;
}

// Every pwm_set_dt() call below (RPC handlers, bring_up_hobby_servo) goes through
// this mutex. Strictly not required by the hardware (each joint owns an
// independent PWM channel/pin, no shared bus to corrupt) but is kept anyway so
// servo[].pos stays consistent when set_positions (RPC thread) and any future
// background task touch it concurrently.
struct k_mutex pwm_bus_mutex;
#endif  // HOBBY_SERVO_ENABLED

Arduino_LED_Matrix matrix;

// set_led_frame() only enqueues a frame here and returns; led_thread (below)
// blocks on this queue and draws at its own pace, so the LED matrix driver can
// never block or wedge the RPC dispatch thread. Depth 1: only the latest frame
// matters, so a still-full queue means the handler just drops the new one.
K_MSGQ_DEFINE(led_frame_queue, sizeof(uint8_t) * LED_MATRIX_PIXELS, 1, 4);

const int LED_THREAD_STACK_SIZE = 1024;
const int LED_THREAD_PRIORITY = 5;  // matches Arduino_RouterBridge's own update thread priority
K_THREAD_STACK_DEFINE(led_thread_stack, LED_THREAD_STACK_SIZE);
struct k_thread led_thread_data;

void led_thread_entry(void *, void *, void *) {
  uint8_t frame[LED_MATRIX_PIXELS];
  while (true) {
    k_msgq_get(&led_frame_queue, frame, K_FOREVER);
    matrix.draw(frame);
  }
}

// Define global array of structs to hold the runtime state for each digit
// joint. joint_id (1-6, see Definitions above) indexes this array. Which fields
// are meaningful for a given joint depends on JOINT_INTERFACE[joint_id].interface
// - a DYNAMIXEL joint uses .pos_/.vel_/.max_vel, a hobby-servo joint uses
// .pwm_index; both use .enabled and .pos.
struct jointState {
  int enabled;      // 0 = disabled (not connected / PWM channel not ready), 1 = enabled
  int pos;          // the current target position (DYNAMIXEL units, or hobby-servo pulse width in us)
#ifdef DYNAMIXEL_ENABLED
  int pos_;         // DYNAMIXEL only: the present position (feedback from servo)
  int vel_;         // DYNAMIXEL only: the present velocity (feedback from servo)
  int min_vel;       // DYNAMIXEL only: the minimum profile velocity (varies from 0-1023)
  int max_vel;       // DYNAMIXEL only: the maximum profile velocity (varies from 0-1023)
#endif
#ifdef HOBBY_SERVO_ENABLED
  size_t pwm_index;  // hobby-servo only: index into pwm_specs[] - which header pin drives this joint
#endif
} joint[DIGIT_NUM + 1];

bool set_positions(std::array<int, DIGIT_NUM> positions);
bool set_led_frame(std::array<uint8_t, LED_MATRIX_PIXELS> led_frame);
std::array<int, 2> get_position(int joint_id);
#ifdef DYNAMIXEL_ENABLED
void bring_up_motor(int i);
#endif
#ifdef HOBBY_SERVO_ENABLED
void bring_up_hobby_servo(int i);
#endif

void setup() {
#ifdef DYNAMIXEL_ENABLED
  k_mutex_init(&dxl_bus_mutex);
#endif
#ifdef HOBBY_SERVO_ENABLED
  k_mutex_init(&pwm_bus_mutex);
#endif

  matrix.begin();

  k_thread_create(&led_thread_data, led_thread_stack, LED_THREAD_STACK_SIZE,
                   led_thread_entry, NULL, NULL, NULL,
                   LED_THREAD_PRIORITY, 0, K_NO_WAIT);
  k_thread_name_set(&led_thread_data, "led");

  Bridge.begin();
  Bridge.provide("move_joint", move_joint);
  Bridge.provide("set_positions", set_positions);
  Bridge.provide("set_led_frame", set_led_frame);
  Bridge.provide("get_position", get_position);

  Serial.begin(9600);
  Serial.println("MyoHandiHand: bringing up digits 1-6...");

#ifdef DYNAMIXEL_ENABLED
  for (int i = 1; i <= DIGIT_NUM; i++) {
    if (JOINT_INTERFACE[i].interface != INTERFACE_DYNAMIXEL) continue;
    joint[i].min_vel = 1;
    joint[i].max_vel = 50;
  }

  // Set Port baudrate to 1Mbps. This has to match with DYNAMIXEL baudrate.
  dxl.begin(1000000);

  // Set Port Protocol Version. This has to match with DYNAMIXEL protocol version.
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  Serial.println("  [dynamixel] pinging digits...");
  delay(100);  // Add delay so there is time for the bus to initialize before starting to ping servos

  // Power-on self test: broadcast ping (DXL_BROADCAST_ID) asks ANY servo on the bus
  // to respond at once, but on this half-duplex-via-resistor bus multiple servos'
  // replies can arrive close enough together to collide and get lost, so it under-
  // reported the real population during bring-up (found 5 of the 6 actual servos).
  // Individually pinging every ID in the sane DYNAMIXEL ID range instead - one at a
  // time, no collision possible - is slower but exhaustive, and is what actually
  // drives joint[i].enabled below (not this scan). This diagnostic scan is purely
  // informational, so a servo at an ID not currently wired into JOINT_INTERFACE[]
  // would still show up here.
  {
    Serial.println("  [dynamixel diag] individually pinging ids 1-20 (collision-free)...");
    uint8_t diag_found = 0;
    for (uint8_t id = 1; id <= 20; id++) {
      if (dxl.ping(id)) {
        Serial.print("    id ");
        Serial.print(id);
        Serial.println(": found");
        diag_found++;
      }
    }
    Serial.print("  [dynamixel diag] total found: ");
    Serial.println(diag_found);
  }
#endif  // DYNAMIXEL_ENABLED

  for (int i = 1; i <= DIGIT_NUM; i++) {
    if (JOINT_INTERFACE[i].interface == INTERFACE_DYNAMIXEL) {
#ifdef DYNAMIXEL_ENABLED
      bool found = dxl.ping(JOINT_INTERFACE[i].dxl_id);
      Serial.print("  joint ");
      Serial.print(i);
      Serial.print(" (dynamixel id ");
      Serial.print(JOINT_INTERFACE[i].dxl_id);
      Serial.println(found ? "): found" : "): NOT FOUND (will keep retrying in the background)");
      if (found) {
        bring_up_motor(i);
      }
#endif
    } else {
#ifdef HOBBY_SERVO_ENABLED
      pin_size_t pin = JOINT_INTERFACE[i].hobby_pin;
      size_t idx = pwm_index_for_pin(pin);
      joint[i].pwm_index = idx;

      Serial.print("  joint ");
      Serial.print(i);
      Serial.print(" (hobby servo, pin D");
      Serial.print(pin);
      Serial.print("): ");

      if (idx == (size_t)-1) {
        Serial.println("NOT FOUND in devicetree pwms list - disabled");
        joint[i].enabled = 0;
        continue;
      }

      bring_up_hobby_servo(i);
      Serial.println(joint[i].enabled ? "ready" : "PWM device not ready - disabled");
#endif
    }
  }
}

#ifdef DYNAMIXEL_ENABLED
// Bring a just-detected DYNAMIXEL servo into position mode and enable it: turn
// torque off to configure EEPROM items, set position control mode, torque back
// on, seed pos/pos_ from the servo's actual present position (so it starts
// stopped instead of snapping), and set its configured profile velocity. Marks
// joint[i].enabled once this all completes. Called both from the initial boot
// scan in setup() and from the background retry loop in loop() for any digit
// not found at boot.
void bring_up_motor(int i) {
  int dxl_id = JOINT_INTERFACE[i].dxl_id;

  k_mutex_lock(&dxl_bus_mutex, K_FOREVER);
  dxl.torqueOff(dxl_id);                     // Turn off torque when configuring items in EEPROM area
  dxl.setOperatingMode(dxl_id, OP_POSITION);  // Set operating mode to position control mode
  dxl.torqueOn(dxl_id);                      // Turn the torque back on
  // Read the latest position and velocity values from the servo
  joint[i].pos_ = dxl.getPresentPosition(dxl_id);
  joint[i].vel_ = dxl.getPresentVelocity(dxl_id);
  // Set the initial target position equal to the current position so the servo starts stopped
  joint[i].pos = joint[i].pos_;
  dxl.setGoalPosition(dxl_id, joint[i].pos);
  dxl.writeControlTableItem(PROFILE_VELOCITY, dxl_id, joint[i].max_vel);
  k_mutex_unlock(&dxl_bus_mutex);
  joint[i].enabled = 1;
}
#endif  // DYNAMIXEL_ENABLED

#ifdef HOBBY_SERVO_ENABLED
// Apply pinctrl for this joint's PWM channel, confirm the underlying PWM device
// is ready, and drive it to center pulse width so an idle/just-powered servo
// doesn't start at some arbitrary or zero duty cycle. Marks joint[i].enabled on
// success. Called once per joint from setup() - a hobby servo's PWM channel is
// either wired up and ready at boot or it isn't (no ping/detect handshake exists
// for a plain PWM signal wire), so there's nothing to retry against in loop().
void bring_up_hobby_servo(int i) {
  size_t idx = joint[i].pwm_index;

  init_dev_apply_channel_pinctrl(pwm_specs[idx].dev,
                                  state_pin_index_from_spec_index(pwm_specs, idx));

  if (!pwm_is_ready_dt(&pwm_specs[idx])) {
    joint[i].enabled = 0;
    return;
  }

  joint[i].pos = SERVO_PULSE_CENTER_US;

  k_mutex_lock(&pwm_bus_mutex, K_FOREVER);
  pwm_set_dt(&pwm_specs[idx], SERVO_PWM_PERIOD_US * NSEC_PER_USEC,
             joint[i].pos * NSEC_PER_USEC);
  k_mutex_unlock(&pwm_bus_mutex);

  joint[i].enabled = 1;
}
#endif  // HOBBY_SERVO_ENABLED

#ifdef DYNAMIXEL_ENABLED
// A DYNAMIXEL digit that doesn't respond at boot (still powering up, loose
// connector, etc.) would otherwise be disabled for the rest of the app's life -
// instead, re-ping any still-disabled DYNAMIXEL joint on a slow interval here so
// it comes online automatically as soon as it responds, without blocking or
// affecting any other joint/RPC handler. Hobby-servo joints have no such retry -
// see bring_up_hobby_servo() above for why.
const unsigned long MOTOR_RETRY_INTERVAL_MS = 2000;
unsigned long last_motor_retry_ms = 0;
#endif

void loop() {
#ifdef DYNAMIXEL_ENABLED
  // Movement itself is entirely event-driven via the RPC handlers below; this is
  // the only thing loop() needs to poll for.
  unsigned long now = millis();
  if (now - last_motor_retry_ms < MOTOR_RETRY_INTERVAL_MS) {
    return;
  }
  last_motor_retry_ms = now;

  for (int i = 1; i <= DIGIT_NUM; i++) {
    if (JOINT_INTERFACE[i].interface != INTERFACE_DYNAMIXEL || joint[i].enabled) {
      continue;
    }
    int dxl_id = JOINT_INTERFACE[i].dxl_id;
    k_mutex_lock(&dxl_bus_mutex, K_FOREVER);
    bool found = dxl.ping(dxl_id);
    k_mutex_unlock(&dxl_bus_mutex);
    if (found) {
      Serial.print("  joint ");
      Serial.print(i);
      Serial.print(" (dynamixel id ");
      Serial.print(dxl_id);
      Serial.println("): found on retry");
      bring_up_motor(i);
    }
  }
#endif  // DYNAMIXEL_ENABLED
  // With no DYNAMIXEL joints configured, movement is entirely event-driven via
  // the RPC handlers below and there's no bus to poll/retry against - loop()
  // has nothing to do.
}

// RPC handler: move a single joint to a goal position, using its configured
// profile velocity for a DYNAMIXEL joint, or clamped to
// SERVO_PULSE_MIN_US/MAX_US for a hobby-servo joint. Returns false if the joint
// id is out of range or that joint isn't enabled yet. Intended for bench
// testing/manual moves - set_positions() is the normal driving path.
bool move_joint(int joint_id, int goal_position) {
  if (joint_id < 1 || joint_id > DIGIT_NUM || !joint[joint_id].enabled) {
    return false;
  }

  if (JOINT_INTERFACE[joint_id].interface == INTERFACE_DYNAMIXEL) {
#ifdef DYNAMIXEL_ENABLED
    int dxl_id = JOINT_INTERFACE[joint_id].dxl_id;
    joint[joint_id].pos = goal_position;

    k_mutex_lock(&dxl_bus_mutex, K_FOREVER);
    dxl.writeControlTableItem(PROFILE_VELOCITY, dxl_id, joint[joint_id].max_vel);
    dxl.setGoalPosition(dxl_id, joint[joint_id].pos);
    k_mutex_unlock(&dxl_bus_mutex);
#endif
  } else {
#ifdef HOBBY_SERVO_ENABLED
    joint[joint_id].pos = clamp_pulse_us(goal_position);

    k_mutex_lock(&pwm_bus_mutex, K_FOREVER);
    pwm_set_dt(&pwm_specs[joint[joint_id].pwm_index], SERVO_PWM_PERIOD_US * NSEC_PER_USEC,
               joint[joint_id].pos * NSEC_PER_USEC);
    k_mutex_unlock(&pwm_bus_mutex);
#endif
  }

  return true;
}

// RPC handler: drive every enabled joint to positions[joint_id - 1], sent as-is
// to a DYNAMIXEL joint (no range checking) or clamped to
// SERVO_PULSE_MIN_US/MAX_US for a hobby-servo joint. Joints not enabled are
// skipped. Non-blocking. Always returns true.
bool set_positions(std::array<int, DIGIT_NUM> positions) {
#ifdef DYNAMIXEL_ENABLED
  k_mutex_lock(&dxl_bus_mutex, K_FOREVER);
  for (int joint_id = 1; joint_id <= DIGIT_NUM; joint_id++) {
    if (!joint[joint_id].enabled || JOINT_INTERFACE[joint_id].interface != INTERFACE_DYNAMIXEL) {
      continue;
    }
    int dxl_id = JOINT_INTERFACE[joint_id].dxl_id;
    joint[joint_id].pos = positions[joint_id - 1];

    dxl.writeControlTableItem(PROFILE_VELOCITY, dxl_id, joint[joint_id].max_vel);
    dxl.setGoalPosition(dxl_id, joint[joint_id].pos);
  }
  k_mutex_unlock(&dxl_bus_mutex);
#endif

#ifdef HOBBY_SERVO_ENABLED
  k_mutex_lock(&pwm_bus_mutex, K_FOREVER);
  for (int joint_id = 1; joint_id <= DIGIT_NUM; joint_id++) {
    if (!joint[joint_id].enabled || JOINT_INTERFACE[joint_id].interface != INTERFACE_HOBBY_SERVO) {
      continue;
    }
    joint[joint_id].pos = clamp_pulse_us(positions[joint_id - 1]);

    pwm_set_dt(&pwm_specs[joint[joint_id].pwm_index], SERVO_PWM_PERIOD_US * NSEC_PER_USEC,
               joint[joint_id].pos * NSEC_PER_USEC);
  }
  k_mutex_unlock(&pwm_bus_mutex);
#endif

  return true;
}

// RPC handler: hand `led_frame` off to led_thread (see above) to draw. Non-blocking
// - drops the frame if led_thread hasn't drained the previous one yet. Always
// returns true.
bool set_led_frame(std::array<uint8_t, LED_MATRIX_PIXELS> led_frame) {
  k_msgq_put(&led_frame_queue, led_frame.data(), K_NO_WAIT);
  return true;
}

// RPC handler: DYNAMIXEL joint - read a joint's live present position and
// velocity straight from the servo (not the last commanded value). Hobby-servo
// joint - hobby servos have no feedback line, so this returns the last
// commanded pulse width instead (state this sketch tracks on the caller's
// behalf, not a live read), and velocity is always 0 (no velocity concept for a
// hobby servo driven this way). Returns [-1, -1] if joint_id is out of range or
// that joint isn't enabled.
std::array<int, 2> get_position(int joint_id) {
  if (joint_id < 1 || joint_id > DIGIT_NUM || !joint[joint_id].enabled) {
    return {-1, -1};
  }

  if (JOINT_INTERFACE[joint_id].interface == INTERFACE_DYNAMIXEL) {
#ifdef DYNAMIXEL_ENABLED
    int dxl_id = JOINT_INTERFACE[joint_id].dxl_id;
    k_mutex_lock(&dxl_bus_mutex, K_FOREVER);
    joint[joint_id].pos_ = dxl.getPresentPosition(dxl_id);
    joint[joint_id].vel_ = dxl.getPresentVelocity(dxl_id);
    k_mutex_unlock(&dxl_bus_mutex);

    return {joint[joint_id].pos_, joint[joint_id].vel_};
#endif
  }

  return {(int)joint[joint_id].pos, 0};
}
