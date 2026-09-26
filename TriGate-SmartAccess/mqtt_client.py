"""
mqtt_client.py - Raspberry Pi <-> ESP32 serial link
Corrected in the TriGate code review - Document 2 of 6.
Full explanation, tests and installation steps: TriGate_Doc2_mqtt_client.pdf
Same class names (GarageSerial / GarageMQTT), same functions and same
arguments as the original, so main.py needs no change to use this file.
Every change is marked with a "FIX n" comment next to the lines it touches:
FIX 1 find the ESP32 by itself (the port name changes after reboots)
FIX 2 open the port without resetting the ESP32 (DTR and RTS low)
FIX 3 close the port without resetting the ESP32 (HUPCL off)
FIX 4 report "connected" only once the board has actually spoken
FIX 5 detect a dead link instead of hiding it
FIX 6 the next command reconnects by itself
FIX 7 send_scanning(False) no longer sends RESET, so the lockout works
FIX 8 send nothing while the gate is locked out
FIX 9 keep the 10-second HEARTBEAT out of the normal log
FIX 10 __init__'s signature was corrupted (a statement was fused into the
       parameter list, so the file could not even be imported); restored
       it to a normal constructor
"""
import serial
import threading
import logging
import time
import glob  # FIX 1
import termios  # FIX 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# FIX 1: find the ESP32 instead of hard-coding /dev/ttyACM0.
# /dev/serial/by-id/ is created by Linux on every boot and never renumbers,
# so the ttyACM0 <-> ttyACM1 flip-flop no longer matters.
def find_esp32_port():
    for pattern in ("/dev/serial/by-id/*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    return None


class GarageSerial:
    # FIX 8: FIX 7 makes the 3-strikes lockout work for the first time. The
    # current firmware stops reading serial during its 60 s lockout, then
    # runs everything it was sent once the lockout ends - a GRANTED sent
    # during the lockout opened the gate about 53 s later, with nobody
    # there. So this client sends nothing (except RESET) while the gate is
    # locked out. 70 s = the firmware's 60 s plus a safe margin.
    LOCKOUT_SECONDS = 70

    # FIX 10: this was originally
    #   def __init__(self, port=None, baud=115200, broker_host=None,
    #                broker_port=1883, on_door_status=self.fixed_port = port
    #   self.port = port
    # i.e. an assignment statement had been fused into the parameter list,
    # which made the file a SyntaxError on import. Restored to a normal
    # signature; the two lines it swallowed are now the first two lines
    # of the body.
    def __init__(self, port=None, baud=115200, broker_host=None, broker_port=1883, on_door_status=None):
        self.fixed_port = port  # FIX 1: None = find it automatically
        self.port = port
        self.baud = baud
        self.on_door_status = on_door_status
        self.ser = None
        self.connected = False
        self._last_reconnect = 0  # FIX 6
        self._lockout_until = 0.0  # FIX 8
        self._denials = 0  # FIX 8

    def connect(self, retries=2):
        for attempt in range(1, retries + 1):
            try:
                self.port = self.fixed_port or find_esp32_port()  # FIX 1
                if not self.port:
                    raise RuntimeError("no serial device found")

                logger.info(f"Connecting to ESP32 on {self.port} (attempt {attempt})")

                # FIX 2: serial.Serial(port, baud) opens with DTR and RTS
                # high. On an ESP32 those lines drive reset and boot mode, so
                # opening the port reset the board - sometimes into its
                # bootloader, where it is "connected" but runs nothing.
                # Set both low BEFORE opening.
                self.ser = serial.Serial()
                self.ser.port = self.port
                self.ser.baudrate = self.baud
                self.ser.timeout = 2
                self.ser.dtr = False
                self.ser.rts = False
                self.ser.open()

                # FIX 3: stop Linux resetting the ESP32 when the port closes.
                attrs = termios.tcgetattr(self.ser.fileno())
                attrs[2] &= ~termios.HUPCL
                termios.tcsetattr(self.ser.fileno(), termios.TCSANOW, attrs)

                # FIX 4: was time.sleep(0.5) then connected = True. The port
                # opens even when the sketch is frozen, so that proved
                # nothing. Wait until the board actually says something
                # (ESP32_READY at boot, HEARTBEAT every 10 s).
                deadline = time.time() + 12
                heard = None
                while time.time() < deadline and not heard:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        heard = line
                if not heard:
                    raise RuntimeError("port opened but the ESP32 sent nothing in 12 s")

                logger.info(f"ESP32 answered: {heard}")
                self.connected = True
                t = threading.Thread(target=self._read_loop, args=(self.ser,), daemon=True)
                t.start()
                logger.info("ESP32 connected via USB")
                return True

            except Exception as e:
                logger.error(f"Connection failed: {e}")
                if self.ser:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                self.ser = None
                time.sleep(1)

        logger.warning("Running WITHOUT ESP32")
        return False

    def disconnect(self):
        self.connected = False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _send(self, command):
        # FIX 8: nothing but RESET while the gate is locked out
        left = self._lockout_until - time.time()
        if left > 0 and command != "RESET":
            logger.warning(f"Gate locked out ({int(left) + 1} s left) — not sending: {command}")
            return False

        if not self.connected or not self.ser:
            # FIX 6: try to get the link back instead of giving up for good
            # (at most once every 10 s so a missing board can't stall main.py).
            if time.time() - self._last_reconnect > 10:
                self._last_reconnect = time.time()
                logger.info("ESP32 link is down - trying to reconnect")
                self.connect(retries=1)
            if not self.connected:
                logger.warning(f"ESP32 not connected — skipping: {command}")
                return False

        try:
            self.ser.write(f"{command}\n".encode())
            self.ser.flush()
            logger.info(f"Sent to ESP32: {command}")
            self._count(command)  # FIX 8
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            self.connected = False  # FIX 5: a failed write means a dead link
            return False

    def _read_loop(self, ser=None):
        # FIX 5: each reader only serves the port it was started for, so an
        # old reader can never share a reconnected port with the new one.
        while self.connected and self.ser is ser:
            try:
                if self.ser and self.ser.in_waiting:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        # FIX 9: a HEARTBEAT every 10 s only proves the link
                        # is alive - keep it out of the normal log
                        if line == "HEARTBEAT":
                            logger.debug(f"ESP32 says: {line}")
                        else:
                            logger.info(f"ESP32 says: {line}")
                        self._track_lockout(line)  # FIX 8
                        if self.on_door_status:
                            self.on_door_status(line)
            except Exception as e:
                # FIX 5: was `except: pass` - it hid every error, so an
                # unplugged board looked connected forever.
                logger.error(f"ESP32 link lost: {e}")
                self.connected = False
            time.sleep(0.05)

    def _count(self, command):
        # FIX 8: count failed attempts exactly like the firmware does. Don't
        # wait for the ESP32 to announce the lockout: the current firmware
        # says LOCKOUT_ACTIVE about 5.5 s after the third DENIED, long enough
        # for a new attempt (and a GRANTED) to slip in.
        if command == "DENIED":
            self._denials += 1
            if self._denials >= 3:
                self._denials = 0
                self._lockout_until = time.time() + self.LOCKOUT_SECONDS
                logger.warning("3 denials - the gate locks out now; commands paused")
        elif command in ("GRANTED", "RESET"):
            self._denials = 0

    def _track_lockout(self, line):
        # FIX 8: follow what the ESP32 reports. The lockout ends only when
        # the ESP32 CONFIRMS it (LOCKOUT_CLEARED or RESET_OK) - not when
        # RESET is sent, because the current firmware cannot hear a RESET
        # until its lockout is over.
        if line.startswith("LOCKOUT_ACTIVE"):
            self._lockout_until = time.time() + self.LOCKOUT_SECONDS
        elif line.startswith(("LOCKOUT_CLEARED", "RESET_OK")):
            self._lockout_until = 0.0
            self._denials = 0

    def send_open(self):
        return self._send("GRANTED")

    def send_deny(self):
        return self._send("DENIED")

    def send_close(self):
        return self._send("CLOSE")

    # FIX 7: was "RESET". main.py calls send_scanning(False) after every
    # attempt, and the firmware's RESET sets failCount back to 0 - so the
    # 3-strikes lockout could never trigger. "IDLE" only stops the yellow LED.
    def send_scanning(self, active=True):
        return self._send("SCANNING" if active else "IDLE")

    def send_reset(self):
        return self._send("RESET")

    def publish_lpr_result(self, passed):
        pass

    def publish_vmmr_result(self, passed):
        pass

    def publish_face_result(self, passed):
        pass

    def _publish(self, topic, message):
        self._send(message)


GarageMQTT = GarageSerial
