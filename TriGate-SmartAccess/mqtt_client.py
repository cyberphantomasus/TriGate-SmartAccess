"""mqtt_client.py -- PATCHED

Same class name, same method names, same call signatures. `main.py` does
not need to change its import. Six defects fixed, nothing redesigned.

  FIX 1  line 11  hard-coded "/dev/ttyACM0" while the kernel renumbers
  FIX 2  line 24  serial.Serial(port, baud) asserts DTR and RTS on open
  FIX 3  line 24  HUPCL left on, so closing the port resets the board
  FIX 4  line 25  time.sleep(0.5) is shorter than the firmware's setup()
  FIX 5  line 26  connected=True because a file descriptor opened
  FIX 6  line 62  bare `except: pass` hid every error forever

The comment you left at line 11 -- "VERIFY THIS FIRST, the port name
flip-flopped between ttyACM0 and ttyACM1" -- was the right diagnosis.
FIX 1 makes it permanent so you never have to check again.
"""

import glob
import logging
import os
import threading
import time

import serial

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# [FIX 1] stable identities, tried in order. The by-id paths are created
# by udev from the device's own descriptors and never renumber.
PORT_PATTERNS = (
    "/dev/trigate-esp",                  # udev symlink, if you install one
    "/dev/serial/by-id/*Espressif*",     # ESP32-S3 native USB-Serial/JTAG
    "/dev/serial/by-id/*JTAG*",
    "/dev/serial/by-id/*CP210*",         # common UART bridges
    "/dev/serial/by-id/*CH340*",
    "/dev/ttyACM*",
    "/dev/ttyUSB*",
)


class GarageSerial:
    def __init__(self, port=None, baud=115200, broker_host=None,
                 broker_port=1883, on_door_status=None, on_sensor_update=None):
        self.port = port                 # [FIX 1] None now means "find it"
        self.baud = baud
        self.on_door_status = on_door_status
        self.ser = None
        self.connected = False
        self.firmware_ready = False
        self._last_rx = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _find_port():
        """[FIX 1] Resolve the port instead of hard-coding an index.

        ttyACM0 -> ttyACM1 renumbering on replug is normal on this
        hardware, which is exactly what your line-11 comment describes.
        """
        for pattern in PORT_PATTERNS:
            hits = sorted(glob.glob(pattern))
            if hits:
                return hits[0]
        return None

    @staticmethod
    def _clear_hupcl(ser):
        """[FIX 3] Stop the kernel lowering the modem lines on close.

        With HUPCL set, every clean shutdown of main.py resets the ESP32.
        """
        try:
            import termios
            fd = ser.fileno()
            attrs = termios.tcgetattr(fd)
            attrs[2] &= ~termios.HUPCL
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception as e:
            logger.debug(f"could not clear HUPCL: {e}")

    def connect(self, retries=3):
        for attempt in range(1, retries + 1):
            try:
                port = self.port or self._find_port()
                if not port:
                    raise RuntimeError("no serial device found "
                                       "(is the ESP32 plugged in?)")
                self.port = port
                logger.info(f"Connecting to ESP32 on {port} (attempt {attempt})")

                # [FIX 2] THE important change.
                # serial.Serial(port, baud) asserts DTR and RTS. On an
                # ESP32-S3 the USB-Serial/JTAG peripheral maps DTR->GPIO0
                # and RTS->EN, so opening the port resets the board -- and
                # some DTR/RTS combinations leave it sitting in the ROM
                # download bootloader: enumerated, openable, running
                # nothing. That is the "works alone, dead on the Pi"
                # symptom.
                self.ser = serial.Serial()
                self.ser.port = port
                self.ser.baudrate = self.baud
                self.ser.timeout = 1.0
                self.ser.write_timeout = 2.0
                self.ser.dtr = False
                self.ser.rts = False
                self.ser.open()
                self._clear_hupcl(self.ser)

                # [FIX 4] your setup() takes >1.8 s on its own: three LED
                # cycles at 600 ms plus the OLED init. 0.5 s meant the
                # first command was fired into a booting chip.
                time.sleep(2.5)
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()

                # [FIX 5] the port opens even when the sketch is frozen, so
                # a successful open() proves nothing. Wait for the board to
                # actually speak. ESP32_READY and HEARTBEAT are already in
                # your firmware -- no protocol change needed.
                if not self._wait_for_board(timeout=12.0):
                    raise RuntimeError("no ESP32_READY or HEARTBEAT "
                                       "-- board is not running your sketch")

                self.connected = True
                self._last_rx = time.time()
                t = threading.Thread(target=self._read_loop, daemon=True)
                t.start()
                logger.info("ESP32 connected via USB")
                return True

            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self._close_quiet()
                time.sleep(1)

        logger.warning("Running WITHOUT ESP32")
        return False

    def _wait_for_board(self, timeout=12.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                raw = self.ser.readline()
            except Exception:
                return False
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            logger.info(f"ESP32 says: {line}")
            if line in ("ESP32_READY", "HEARTBEAT", "RESET_OK",
                        "OLED FAILED - continuing without display"):
                self.firmware_ready = True
                return True
        return False

    def _close_quiet(self):
        self.connected = False
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def disconnect(self):
        self._close_quiet()

    # ------------------------------------------------------------------
    def _send(self, command):
        if not self.connected or not self.ser:
            logger.warning(f"ESP32 not connected — skipping: {command}")
            return False
        try:
            self.ser.write(f"{command}\n".encode())
            self.ser.flush()
            logger.info(f"Sent to ESP32: {command}")
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            self.connected = False        # [FIX 6] a dead write means a
            return False                  # dead link; say so

    def _read_loop(self):
        while self.connected:
            try:
                raw = self.ser.readline() if self.ser else b""
                if raw:
                    self._last_rx = time.time()
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if line:
                        logger.info(f"ESP32 says: {line}")
                        if self.on_door_status:
                            self.on_door_status(line)
                # your firmware emits HEARTBEAT every 10 s
                elif self._last_rx and time.time() - self._last_rx > 30:
                    logger.error("no heartbeat for 30 s — link is dead")
                    self.connected = False
            except Exception as e:
                # [FIX 6] was `except: pass`, which also swallowed
                # KeyboardInterrupt and made a dead port look healthy
                # forever. Now a broken link is visible and recoverable.
                logger.error(f"Read failed: {e}")
                self.connected = False
            time.sleep(0.01)

    def healthy(self):
        """New. Lets main.py tell a live link from a dead one."""
        return (self.connected and self.ser is not None
                and self.ser.is_open
                and time.time() - self._last_rx < 30)

    def reconnect(self):
        self._close_quiet()
        return self.connect(retries=1)

    # -- unchanged public API ------------------------------------------
    def send_open(self):     return self._send("GRANTED")
    def send_deny(self):     return self._send("DENIED")
    def send_close(self):    return self._send("CLOSE")
    def send_scanning(self, active=True):
        return self._send("SCANNING" if active else "RESET")

    # your dead stubs, kept so nothing that imports them breaks
    def publish_lpr_result(self, passed): pass
    def publish_vmmr_result(self, passed): pass
    def publish_face_result(self, passed): pass
    def _publish(self, topic, message): self._send(message)


GarageMQTT = GarageSerial
