#!/usr/bin/env python3
"""Screenshot tool for the agent's own verification loop.

Grabs a window (or a monitor / the whole desktop) off the Windows desktop, optionally
after a settle delay, and writes a PNG. It reports per-channel statistics so a caller
can tell "I captured a real frame" from "I captured a black frame" without eyeballing it.

Why a desktop grab instead of PrintWindow: Minecraft runs borderless-windowed here, so
BitBlt of the desktop region is faithful. PrintWindow on a Vulkan swapchain generally
returns black, so it is deliberately not offered.

Usage examples
--------------
    python shot.py --list
    python shot.py --window Minecraft --out shot.png
    python shot.py --window Minecraft --focus --delay 1.5 --stats
    python shot.py --screen --grid 100 --out-dir out
    python shot.py --window Minecraft --repeat 4 --interval 0.25 --json

Exit codes: 0 ok, 2 grabbed but the frame looks empty/black, 3 no matching window,
4 bad arguments.
"""

import argparse
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes

from PIL import Image, ImageDraw, ImageFont, ImageGrab

# Window titles here are often CJK, and a piped stdout defaults to the ANSI code page on
# Windows, which mangles them. Force UTF-8 so callers (and the agent) always get real text.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# --------------------------------------------------------------------------------------
# Win32 bindings
# --------------------------------------------------------------------------------------

user32 = ctypes.WinDLL("user32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)

DWMWA_EXTENDED_FRAME_BOUNDS = 9
SW_RESTORE = 9


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def set_dpi_awareness():
    """Make every rect we read back be in physical pixels.

    Try the modern per-monitor-v2 context first; fall back to the flat system-DPI call.
    A failure here is not fatal, it only means coordinates may be virtualised on a
    scaled display, so the caller's rects would be wrong -- worth knowing about.
    """
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor-v2"
    except AttributeError:
        pass
    try:
        if user32.SetProcessDPIAware():
            return "system"
    except AttributeError:
        pass
    return "none"


WINFUNCTYPE = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)


def enumerate_windows():
    """Every top-level window with a non-empty title."""
    out = []

    def cb(hwnd, _):
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            rect = RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            out.append({
                "hwnd": int(hwnd),
                "title": buf.value,
                "pid": int(pid.value),
                "visible": bool(user32.IsWindowVisible(hwnd)),
                "minimized": bool(user32.IsIconic(hwnd)),
                "rect": [rect.left, rect.top, rect.right, rect.bottom],
                "width": rect.right - rect.left,
                "height": rect.bottom - rect.top,
                "foreground": int(hwnd) == int(user32.GetForegroundWindow()),
            })
        return True

    user32.EnumWindows(WINFUNCTYPE(cb), 0)
    return out


def resolve_window(args):
    """Pick the target window for --window / --hwnd / --pid / --foreground."""
    windows = enumerate_windows()

    if args.hwnd is not None:
        for w in windows:
            if w["hwnd"] == args.hwnd:
                return w
        raise LookupError("no window with hwnd=%d" % args.hwnd)

    if args.pid is not None:
        matches = [w for w in windows if w["pid"] == args.pid and w["visible"]]
        if not matches:
            raise LookupError("no visible window owned by pid=%d" % args.pid)
        return max(matches, key=lambda w: w["width"] * w["height"])

    if args.foreground:
        hwnd = int(user32.GetForegroundWindow())
        for w in windows:
            if w["hwnd"] == hwnd:
                return w
        raise LookupError("foreground window %d has no title" % hwnd)

    if args.window:
        needle = args.window.lower()
        matches = [w for w in windows
                   if w["visible"] and needle in w["title"].lower()]
        if not matches:
            near = [w["title"] for w in windows if w["visible"]][:25]
            raise LookupError("no visible window title contains %r; visible titles: %s"
                              % (args.window, near))
        # Prefer the biggest match: a splash/launcher window often shares the title prefix.
        return max(matches, key=lambda w: w["width"] * w["height"])

    return None


def target_box(args, win):
    """Screen-space box to grab, following --frame."""
    if win is None:
        return None  # whole desktop

    if args.frame == "window":
        r = RECT()
        user32.GetWindowRect(win["hwnd"], ctypes.byref(r))
        return r.left, r.top, r.right, r.bottom

    if args.frame == "extended":
        r = RECT()
        hr = dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(win["hwnd"]), wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(r), ctypes.sizeof(r))
        if hr == 0 and r.right > r.left and r.bottom > r.top:
            return r.left, r.top, r.right, r.bottom
        # Fall through to the plain window rect when DWM refuses (e.g. cloaked window).
        return target_box(argparse.Namespace(frame="window"), win)

    # default: the client area -- what the app actually renders into, no invisible borders.
    client = RECT()
    if not user32.GetClientRect(win["hwnd"], ctypes.byref(client)):
        raise OSError("GetClientRect failed")
    origin = POINT(0, 0)
    if not user32.ClientToScreen(win["hwnd"], ctypes.byref(origin)):
        raise OSError("ClientToScreen failed")
    return origin.x, origin.y, origin.x + client.right, origin.y + client.bottom


def monitor_boxes():
    """Per-monitor screen boxes, left-to-right then top-to-bottom."""
    boxes = []
    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM)

    def cb(hmon, _hdc, _rect, _data):
        info = ctypes.c_ulong()
        rect = RECT()
        # MONITORINFO = cbSize + rcMonitor + rcWork + dwFlags
        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                        ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            boxes.append((mi.rcMonitor.left, mi.rcMonitor.top,
                          mi.rcMonitor.right, mi.rcMonitor.bottom))
        return True

    user32.EnumDisplayMonitors(None, None, MONITORENUMPROC(cb), 0)
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# --------------------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------------------

def overlay_grid(img, step, label_step=None):
    """Draw a coordinate grid. Labels are in image-local pixels, which is what the
    agent needs when it later injects a click at a computed position."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    label_step = label_step or step * 2
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for x in range(step, w, step):
        major = (x % label_step == 0)
        draw.line([(x, 0), (x, h)], fill=(255, 0, 220) if major else (110, 0, 95), width=1)
    for y in range(step, h, step):
        major = (y % label_step == 0)
        draw.line([(0, y), (w, y)], fill=(255, 0, 220) if major else (110, 0, 95), width=1)

    for x in range(label_step, w, label_step):
        for y in range(label_step, h, label_step):
            text = "%d,%d" % (x, y)
            # Outline first so the label stays readable over any content.
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.text((x + 2 + dx, y + 2 + dy), text, fill=(0, 0, 0), font=font)
            draw.text((x + 2, y + 2), text, fill=(255, 240, 0), font=font)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(255, 0, 220), width=2)
    return img


def draw_marks(img, marks, annotate):
    draw = ImageDraw.Draw(img)
    for spec in marks:
        try:
            x, y = (int(v) for v in spec.split(","))
        except ValueError:
            raise SystemExit("--mark wants x,y (got %r)" % spec)
        r = 9
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(0, 255, 80), width=2)
        draw.line([(x - r * 2, y), (x + r * 2, y)], fill=(0, 255, 80), width=1)
        draw.line([(x, y - r * 2), (x, y + r * 2)], fill=(0, 255, 80), width=1)
        draw.text((x + r + 3, y - r - 3), "%d,%d" % (x, y), fill=(0, 255, 80))
    if annotate:
        draw.rectangle([0, 0, img.width - 1, 20], fill=(0, 0, 0))
        draw.text((4, 4), annotate[:120], fill=(255, 255, 0))
    return img


def image_stats(img):
    """Enough numbers to distinguish a real frame from a blank/black grab."""
    gray = img.convert("L")
    hist = gray.histogram()
    total = sum(hist) or 1
    mean = sum(i * c for i, c in enumerate(hist)) / total
    variance = sum(c * (i - mean) ** 2 for i, c in enumerate(hist)) / total
    dark = sum(hist[:8]) / total
    bright = sum(hist[248:]) / total
    extrema = gray.getextrema()
    # distinct colours: sample-subsample so a 4K grab does not cost a second. Slice the raw
    # buffer rather than getdata(), which is deprecated in newer Pillow.
    small = img.convert("RGB").resize((min(img.width, 256), min(img.height, 256)))
    buf = small.tobytes()
    distinct = len({buf[i:i + 3] for i in range(0, len(buf), 3)})
    return {
        "mean_luma": round(mean, 3),
        "std_luma": round(variance ** 0.5, 3),
        "min_luma": extrema[0],
        "max_luma": extrema[1],
        "frac_dark_le7": round(dark, 5),
        "frac_bright_ge248": round(bright, 5),
        "distinct_colors_256": distinct,
    }


def looks_blank(stats):
    """A frame that is essentially uniform. Renderers that failed to draw, a minimized
    window, or an occluded/blacklisted swapchain all land here."""
    return (stats["frac_dark_le7"] > 0.995 and stats["distinct_colors_256"] <= 2) \
        or (stats["std_luma"] < 0.75 and stats["distinct_colors_256"] <= 2)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Grab a window / monitor / desktop to PNG, with sanity statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    target = p.add_argument_group("target")
    target.add_argument("--window", metavar="SUBSTR",
                        help="visible window whose title contains SUBSTR (case-insensitive)")
    target.add_argument("--hwnd", type=lambda v: int(v, 0), help="exact window handle")
    target.add_argument("--pid", type=int, help="largest visible window of this process id")
    target.add_argument("--foreground", action="store_true", help="the current foreground window")
    target.add_argument("--screen", action="store_true",
                        help="whole virtual desktop (default when no target is given)")
    target.add_argument("--monitor", type=int, metavar="N",
                        help="a single monitor, 0-based, left-to-right")
    target.add_argument("--frame", choices=("client", "window", "extended"), default="client",
                        help="which rect of a window to grab (default: client area)")
    target.add_argument("--list", action="store_true", help="list window titles and exit")

    timing = p.add_argument_group("timing")
    timing.add_argument("--delay", type=float, default=0.0,
                        help="seconds to wait after --focus before grabbing (render settle)")
    timing.add_argument("--repeat", type=int, default=1, help="grab N frames")
    timing.add_argument("--interval", type=float, default=0.2,
                        help="seconds between repeated grabs")

    output = p.add_argument_group("output")
    output.add_argument("--out", metavar="PATH", help="output PNG path")
    output.add_argument("--out-dir", metavar="DIR", default=".",
                        help="directory for auto-named output (with --repeat)")
    output.add_argument("--tag", default="", help="string inserted into auto-generated names")
    output.add_argument("--scale", type=float, default=1.0,
                        help="resize factor applied before saving")
    output.add_argument("--crop", metavar="X,Y,W,H", help="crop (image-local, after grab)")

    view = p.add_argument_group("annotation")
    view.add_argument("--grid", type=int, metavar="STEP",
                      help="overlay a coordinate grid with STEP-pixel cells")
    view.add_argument("--mark", action="append", default=[], metavar="X,Y",
                      help="draw a crosshair marker; repeatable")
    view.add_argument("--annotate", default="", help="caption drawn in the top-left corner")

    misc = p.add_argument_group("misc")
    misc.add_argument("--focus", action="store_true",
                      help="restore + raise the window before grabbing")
    misc.add_argument("--stats", action="store_true", help="print image statistics")
    misc.add_argument("--json", action="store_true", help="print one JSON object on stdout")
    misc.add_argument("--quiet", action="store_true", help="suppress the human summary")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    dpi = set_dpi_awareness()

    if args.list:
        rows = [w for w in enumerate_windows() if w["visible"]]
        rows.sort(key=lambda w: -(w["width"] * w["height"]))
        for w in rows:
            print("%10d  pid=%-6d %5dx%-5d %s%s"
                  % (w["hwnd"], w["pid"], w["width"], w["height"],
                     "[fg] " if w["foreground"] else "     ", w["title"][:80]))
        return 0

    if args.monitor is not None:
        boxes = monitor_boxes()
        if not 0 <= args.monitor < len(boxes):
            print("monitor %d out of range (found %d)" % (args.monitor, len(boxes)),
                  file=sys.stderr)
            return 4
        box = boxes[args.monitor]
        win = None
    elif args.window or args.hwnd is not None or args.pid is not None or args.foreground:
        try:
            win = resolve_window(args)
        except LookupError as exc:
            print("ERROR: %s" % exc, file=sys.stderr)
            return 3
        box = target_box(args, win)
    else:
        win = None
        box = None  # whole desktop

    if args.focus and win:
        if win["minimized"]:
            user32.ShowWindow(wintypes.HWND(win["hwnd"]), SW_RESTORE)
        user32.SetForegroundWindow(wintypes.HWND(win["hwnd"]))

    if args.delay:
        time.sleep(args.delay)

    results = []
    for i in range(max(1, args.repeat)):
        if i:
            time.sleep(max(0.0, args.interval))

        if box is None:
            img = ImageGrab.grab(all_screens=True)
        else:
            left, top, right, bottom = box
            if right - left < 8 or bottom - top < 8:
                print("ERROR: degenerate grab box %dx%d" % (right - left, bottom - top),
                      file=sys.stderr)
                return 4
            img = ImageGrab.grab(bbox=box, all_screens=True)

        img = img.convert("RGB")

        if args.crop:
            try:
                cx, cy, cw, ch = (int(v) for v in args.crop.split(","))
            except ValueError:
                print("--crop wants X,Y,W,H", file=sys.stderr)
                return 4
            img = img.crop((cx, cy, cx + cw, cy + ch))

        if args.scale and args.scale != 1.0:
            img = img.resize((max(1, round(img.width * args.scale)),
                              max(1, round(img.height * args.scale))), Image.LANCZOS)

        if args.grid:
            img = overlay_grid(img, args.grid)
        if args.mark or args.annotate:
            img = draw_marks(img, args.mark, args.annotate)

        stats = image_stats(img)

        if args.repeat > 1:
            os.makedirs(args.out_dir, exist_ok=True)
            parts = ["shot", args.tag or (args.window or "screen"),
                     time.strftime("%Y%m%d-%H%M%S")]
            path = os.path.join(args.out_dir, "%s-%02d.png" % ("-".join(p for p in parts if p), i))
        elif args.out:
            path = args.out
        else:
            os.makedirs(args.out_dir, exist_ok=True)
            parts = ["shot", args.tag or (args.window or "screen"),
                     time.strftime("%Y%m%d-%H%M%S")]
            path = os.path.join(args.out_dir, "%s.png" % "-".join(p for p in parts if p))

        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        img.save(path)

        results.append({"path": os.path.abspath(path), "width": img.width,
                        "height": img.height, "stats": stats,
                        "blank": looks_blank(stats)})

    payload = {
        "dpi_awareness": dpi,
        "window": None if win is None else
        {"hwnd": win["hwnd"], "title": win["title"], "pid": win["pid"]},
        "frame": args.frame if win else ("monitor" if args.monitor is not None else "desktop"),
        "box": None if box is None else list(box),
        "grabs": results,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    elif not args.quiet:
        for r in results:
            s = r["stats"]
            print("IMAGE -> %s  (%dx%d)  dpi=%s" % (r["path"], r["width"], r["height"], dpi))
            print("   luma mean=%.2f std=%.2f min=%d max=%d  dark=%.4f bright=%.4f colors=%d%s"
                  % (s["mean_luma"], s["std_luma"], s["min_luma"], s["max_luma"],
                     s["frac_dark_le7"], s["frac_bright_ge248"],
                     s["distinct_colors_256"], "   <== BLANK/BLACK" if r["blank"] else ""))

    return 2 if any(r["blank"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
