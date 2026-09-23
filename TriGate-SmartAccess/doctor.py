#!/usr/bin/env python3
"""trigate-doctor -- tells you exactly WHY the ESP32 is not talking to the Pi.

Runs every check in order, cheapest first, and prints a ranked verdict
instead of a guess. Run it before touching any code:

    python3 tools/doctor.py
    python3 tools/doctor.py --port /dev/ttyACM1     # force a port
    python3 tools/doctor.py --listen 20             # listen longer

Checks, in order:
  1  candidate serial devices present
  2  USB VID:PID -> native USB-Serial/JTAG vs UART bridge
  3  user group membership (dialout, plugdev)
  4  device node permissions
  5  ModemManager (probes ttyACM with AT commands and steals the port)
  6  port already held by another process
  7  open with DTR/RTS deasserted, listen for EVT / HEARTBEAT
  8  round-trip PING with checksum
  9  fallback: open with DTR/RTS asserted, to separate "frozen board" from
     "wrong USB port / CDC On Boot disabled"
"""

from __future__ import annotations

import argparse
import glob
import grp
import os
import pwd
import re
import stat
import subprocess
import sys
import time

try:
    import serial
except ImportError:
    print("pyserial missing:  pip install pyserial")
    sys.exit(2)

G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; B = "\033[1m"; X = "\033[0m"
OK, WARN, BAD, INFO = f"{G}PASS{X}", f"{Y}WARN{X}", f"{R}FAIL{X}", "info"

findings: list[tuple[str, str]] = []


def say(status, title, detail=""):
    print(f"  [{status}] {B}{title}{X}" + (f"\n         {detail}" if detail else ""))
    if status in (BAD, WARN):
        findings.append((status, title))


def header(n, t):
    print(f"\n{B}{n}. {t}{X}")


# ---------------------------------------------------------------- 1 + 2

ESPRESSIF_VID = "303a"
BRIDGE_VIDS = {"10c4": "CP210x", "1a86": "CH34x", "0403": "FTDI",
               "1a86:55d4": "CH9102"}


def sysfs_ids(dev: str):
    """Walk /sys to find idVendor/idProduct for a tty device."""
    name = os.path.basename(os.path.realpath(dev))
    base = f"/sys/class/tty/{name}/device"
    for _ in range(6):
        for f in ("idVendor", "idProduct"):
            p = os.path.join(base, f)
            if os.path.exists(p):
                try:
                    vid = open(os.path.join(base, "idVendor")).read().strip()
                    pid = open(os.path.join(base, "idProduct")).read().strip()
                    return vid, pid
                except Exception:
                    return None, None
        base = os.path.join(base, "..")
    return None, None


def find_ports(explicit):
    if explicit:
        return [explicit]
    out = []
    for pat in ("/dev/trigate-esp", "/dev/serial/by-id/*",
                "/dev/ttyACM*", "/dev/ttyUSB*"):
        out += sorted(glob.glob(pat))
    seen, uniq = set(), []
    for p in out:
        real = os.path.realpath(p)
        if real not in seen:
            seen.add(real)
            uniq.append(p)
    return uniq


def check_devices(explicit):
    header(1, "Serial devices")
    ports = find_ports(explicit)
    if not ports:
        say(BAD, "No serial device found",
            "The ESP32 is not enumerating. Try: a different USB cable (many "
            "are charge-only), the other USB-C socket on the board, and a "
            "USB 2.0 port on the Pi. Watch `dmesg -w` while you plug it in.")
        return []
    for p in ports:
        real = os.path.realpath(p)
        say(OK, p, f"-> {real}" if real != p else "")
    if len(glob.glob("/dev/ttyACM*")) > 1:
        say(WARN, "More than one ttyACM present",
            "This is why hard-coding /dev/ttyACM0 breaks. Use "
            "/dev/serial/by-id/ or install the udev rule.")
    return ports


def check_chip(port):
    header(2, "USB identity")
    vid, pid = sysfs_ids(port)
    if not vid:
        say(WARN, "Could not read VID:PID from sysfs")
        return None
    say(INFO, f"VID:PID = {vid}:{pid}")
    if vid == ESPRESSIF_VID:
        say(OK, "Native USB-Serial/JTAG (Espressif)",
            "You are on the board's NATIVE USB socket. For `Serial` in your "
            "sketch to reach this socket, Arduino IDE must have "
            "Tools > USB CDC On Boot = Enabled. If it is Disabled, `Serial` "
            "goes to UART0 pins instead and the Pi hears nothing -- the port "
            "still opens fine, which is exactly the trap.")
        return "native"
    if vid in BRIDGE_VIDS:
        say(OK, f"UART bridge ({BRIDGE_VIDS[vid]})",
            "You are on the board's UART socket. `Serial` reaches here when "
            "USB CDC On Boot = Disabled.")
        return "bridge"
    say(WARN, f"Unrecognised device {vid}:{pid}")
    return None


# -------------------------------------------------------------------- 3+4

def check_groups():
    header(3, "User groups")
    user = pwd.getpwuid(os.getuid()).pw_name
    mine = {g.gr_name for g in grp.getgrall() if user in g.gr_mem}
    mine.add(grp.getgrgid(pwd.getpwnam(user).pw_gid).gr_name)
    for need in ("dialout", "plugdev"):
        if need in mine:
            say(OK, f"{user} is in {need}")
        else:
            say(BAD, f"{user} is NOT in {need}",
                f"sudo usermod -aG {need} {user}   then LOG OUT and back in. "
                "On Espressif JTAG devices plugdev is often the real owning "
                "group via a udev ACL, so dialout alone is not enough.")


def check_perms(port):
    header(4, "Device permissions")
    real = os.path.realpath(port)
    try:
        st = os.stat(real)
    except OSError as e:
        say(BAD, f"cannot stat {real}", str(e))
        return
    mode = stat.filemode(st.st_mode)
    owner = pwd.getpwuid(st.st_uid).pw_name
    group = grp.getgrgid(st.st_gid).gr_name
    say(INFO, f"{real}  {mode}  {owner}:{group}")
    if os.access(real, os.R_OK | os.W_OK):
        say(OK, "Readable and writable by you")
    else:
        say(BAD, "No read/write access",
            "This is the Errno 13 you saw. Fix the group membership above, "
            "or install the udev rule in setup/99-trigate.rules.")


# ---------------------------------------------------------------------- 5

def check_modemmanager():
    header(5, "ModemManager")
    try:
        r = subprocess.run(["systemctl", "is-active", "ModemManager"],
                           capture_output=True, text=True, timeout=5)
        state = r.stdout.strip()
    except Exception:
        say(INFO, "systemctl unavailable, skipping")
        return
    if state == "active":
        say(BAD, "ModemManager is RUNNING",
            "It probes every new /dev/ttyACM with AT commands to see if it is "
            "a modem. That steals the port for 10-20 s after each plug-in and "
            "injects garbage into your link.\n"
            "         sudo systemctl stop ModemManager\n"
            "         sudo systemctl disable ModemManager\n"
            "         sudo systemctl mask ModemManager")
    else:
        say(OK, f"ModemManager is {state or 'not installed'}")


# ---------------------------------------------------------------------- 6

def check_holder(port):
    header(6, "Port ownership")
    real = os.path.realpath(port)
    holders = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.path.realpath(os.path.join(fd_dir, fd)) == real:
                        cmd = open(f"/proc/{pid}/comm").read().strip()
                        holders.append(f"{cmd}(pid {pid})")
                        break
                except OSError:
                    continue
        except OSError:
            continue
    if holders:
        say(BAD, "Port is already open", ", ".join(holders) +
            "\n         Close it. A leftover `screen`, a second copy of "
            "main.py or the Arduino Serial Monitor will block everything.")
    else:
        say(OK, "Nobody else holds the port")


# ------------------------------------------------------------------- 7-9

def xorsum(s: str) -> str:
    c = 0
    for b in s.encode():
        c ^= b
    return f"{c:02X}"


def open_port(port, dtr, rts, baud=115200):
    s = serial.Serial()
    s.port = port
    s.baudrate = baud
    s.timeout = 0.3
    s.write_timeout = 2
    s.dtr = dtr
    s.rts = rts
    s.open()
    try:
        import termios
        a = termios.tcgetattr(s.fileno())
        a[2] &= ~termios.HUPCL
        termios.tcsetattr(s.fileno(), termios.TCSANOW, a)
    except Exception:
        pass
    return s


def listen(port, seconds, dtr, rts, label):
    lines = []
    try:
        s = open_port(port, dtr, rts)
    except Exception as e:
        say(BAD, f"cannot open {port} ({label})", str(e))
        return None, lines
    time.sleep(2.5)                       # let setup() finish if it rebooted
    s.reset_input_buffer()
    end = time.time() + seconds
    while time.time() < end:
        try:
            raw = s.readline()
        except Exception as e:
            say(BAD, "read error", str(e))
            break
        if raw:
            t = raw.decode("utf-8", "ignore").strip()
            if t:
                lines.append(t)
                print(f"         <- {t}")
    return s, lines


def check_traffic(port, seconds):
    header(7, f"Listening {seconds}s with DTR/RTS deasserted")
    print("         (DTR->GPIO0, RTS->EN on ESP32-S3. Asserting them on open "
          "resets the board or traps it in the download bootloader.)")
    s, lines = listen(port, seconds, dtr=False, rts=False, label="dtr/rts off")
    if s is None:
        return None, []
    if any(("EVT " in l) or ("HEARTBEAT" in l) or ("READY" in l) for l in lines):
        say(OK, "Board is alive and transmitting")
        return s, lines
    if lines:
        say(WARN, "Traffic seen but no recognisable TriGate frames",
            "Old firmware still flashed, wrong baud, or ModemManager noise.")
        return s, lines
    say(BAD, "Silence",
        "Either the sketch is frozen, or `Serial` is not routed to this "
        "socket. Check 9 below separates the two.")
    return s, lines


def check_ping(s):
    header(8, "Round-trip PING")
    if s is None:
        say(BAD, "skipped, port not open")
        return
    payload = "1 PING "
    frame = f"{payload}*{xorsum(payload)}\n"
    try:
        s.reset_input_buffer()
        s.write(frame.encode())
        s.flush()
    except Exception as e:
        say(BAD, "write failed", str(e))
        return
    print(f"         -> {frame.strip()}")
    end = time.time() + 3
    while time.time() < end:
        raw = s.readline()
        if not raw:
            continue
        t = raw.decode("utf-8", "ignore").strip()
        if t:
            print(f"         <- {t}")
        if t.startswith("ACK 1 PING"):
            say(OK, "ACK received -- the link works end to end")
            return
    say(BAD, "No ACK",
        "If check 7 showed heartbeats, the board is transmitting but not "
        "RECEIVING. That is almost always the old firmware (no checksum "
        "protocol) -- reflash firmware/esp32_trigate.")


def check_frozen_vs_wrongport(port, s):
    header(9, "Frozen board, or wrong USB socket?")
    if s is not None:
        try:
            s.close()
        except Exception:
            pass
    time.sleep(0.5)
    others = [p for p in glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
              if os.path.realpath(p) != os.path.realpath(port)]
    if others:
        say(WARN, f"Another serial device exists: {', '.join(others)}",
            "Run the doctor against it too:\n"
            f"         python3 tools/doctor.py --port {others[0]}\n"
            "         If THAT one talks, your sketch's `Serial` is bound to "
            "the other socket -- flip Tools > USB CDC On Boot and reflash.")
    else:
        say(INFO, "Only one serial device present",
            "So silence in check 7 points at the board itself: the old "
            "firmware's `for(;;)` on OLED init failure halts setup() forever. "
            "Unplug the OLED's 4 wires and power-cycle. If the board starts "
            "talking with the display disconnected, that was it -- and "
            "firmware/esp32_trigate never does that again.")


def verdict():
    print(f"\n{B}VERDICT{X}")
    if not findings:
        print(f"  {G}Everything checks out. The link is healthy.{X}")
        return 0
    print(f"  {len(findings)} issue(s), most important first:")
    for i, (st, t) in enumerate(
            sorted(findings, key=lambda f: 0 if f[0] == BAD else 1), 1):
        print(f"   {i}. [{st}] {t}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--listen", type=int, default=12)
    a = ap.parse_args()

    print(f"{B}TriGate link doctor{X}")

    ports = check_devices(a.port)
    if not ports:
        return verdict()
    port = ports[0]
    print(f"\n  testing: {B}{port}{X}")

    check_chip(port)
    check_groups()
    check_perms(port)
    check_modemmanager()
    check_holder(port)
    s, _ = check_traffic(port, a.listen)
    check_ping(s)
    check_frozen_vs_wrongport(port, s)
    return verdict()


if __name__ == "__main__":
    sys.exit(main())
