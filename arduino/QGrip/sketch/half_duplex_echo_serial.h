// SPDX-License-Identifier: MPL-2.0
//
// HalfDuplexEchoSerial: a HardwareSerial decorator for a single-wire DYNAMIXEL bus
// wired directly to the motor with no transceiver/adapter circuit (TX and RX tied
// together on one physical wire, no DXL_DIR_PIN).
//
// On such a wire, every byte the MCU transmits is also immediately looped back into
// its own RX, arriving before the servo's real reply. Dynamixel2Arduino's packet
// parser starts consuming RX bytes right after a write with no built-in echo
// discard, so left alone it would have to parse-and-reject the echoed instruction
// packet before it could find the real status packet - fragile to rely on for a
// device that moves a hand. This wrapper counts bytes as they're written and
// transparently swallows that many bytes back out of the underlying Serial's RX
// stream before exposing anything to callers (i.e. to the Dynamixel2Arduino
// library), so the library only ever sees the servo's actual response.
//
// This assumes standard half-duplex electrical wiring: Serial3 TX and RX both tied
// to the DYNAMIXEL data line, with a small series resistor (~150-470 ohm) on TX to
// avoid driver contention while RX is also actively driven by the bus/servo.

#pragma once

#include <Arduino.h>
#include <Arduino_RouterBridge.h>

// Set to 1 temporarily to trace every byte written/dropped/passed-through on
// Serial (the Bridge Monitor) - verbose, only for diagnosing the echo guard
// itself. Leave at 0 for normal operation.
#define HALF_DUPLEX_ECHO_SERIAL_DEBUG 0

class HalfDuplexEchoSerial : public HardwareSerial {
public:
  explicit HalfDuplexEchoSerial(HardwareSerial &port) : port_(port) {}

  void begin(unsigned long baud) override {
    port_.begin(baud);
    pending_echo_ = 0;
  }

  void begin(unsigned long baud, uint16_t config) override {
    port_.begin(baud, config);
    pending_echo_ = 0;
  }

  void end() override {
    port_.end();
  }

  size_t write(uint8_t b) override {
    size_t n = port_.write(b);
    if (n) {
      pending_echo_ += n;
#if HALF_DUPLEX_ECHO_SERIAL_DEBUG
      Serial.print("[dxl tx] 0x");
      Serial.println(b, HEX);
#endif
    }
    return n;
  }

  size_t write(const uint8_t *buffer, size_t size) override {
    size_t n = port_.write(buffer, size);
    pending_echo_ += n;
#if HALF_DUPLEX_ECHO_SERIAL_DEBUG
    Serial.print("[dxl tx] ");
    Serial.print(n);
    Serial.print(" bytes:");
    for (size_t i = 0; i < n; i++) {
      Serial.print(" 0x");
      Serial.print(buffer[i], HEX);
    }
    Serial.println();
#endif
    return n;
  }

  int available() override {
    dropPendingEcho();
    return port_.available();
  }

  int peek() override {
    dropPendingEcho();
    return port_.peek();
  }

  int read() override {
    dropPendingEcho();
    int c = port_.read();
#if HALF_DUPLEX_ECHO_SERIAL_DEBUG
    if (c >= 0) {
      Serial.print("[dxl rx, passed through] 0x");
      Serial.println(c, HEX);
    }
#endif
    return c;
  }

  void flush() override {
    port_.flush();
  }

  operator bool() override {
    return (bool)port_;
  }

private:
  // Discard bytes we ourselves just transmitted before they can reach the caller.
  // Only pulls what's already arrived - never blocks - so a still-in-flight echo
  // byte is dropped the next time available()/read()/peek() is polled instead.
  void dropPendingEcho() {
    while (pending_echo_ > 0 && port_.available() > 0) {
      int c = port_.read();
      pending_echo_--;
#if HALF_DUPLEX_ECHO_SERIAL_DEBUG
      Serial.print("[dxl rx, dropped as echo] 0x");
      Serial.println(c, HEX);
#endif
    }
  }

  HardwareSerial &port_;
  size_t pending_echo_ = 0;
};
