# shot

A screenshot tool for the agent's verification loop: grab a window, a monitor or the whole
desktop to a PNG, with enough instrumentation that "did I actually capture a frame" is
answered by a number instead of by squinting.

```bat
python shot.py --list
python shot.py --window Minecraft --focus --delay 1.5 --stats
python shot.py --foreground --grid 200 --mark 640,360 --out check.png
python shot.py --window Minecraft --repeat 4 --interval 0.25 --json
```

## Why these options exist

* **`--delay`** — a renderer needs a frame or two after a window is raised before what is on
  screen is what you want to look at. Grabbing immediately after `--focus` is how you get a
  shot of the previous frame.
* **`--stats` / `--json`** — reports mean/std/min/max luma, the fraction of near-black and
  near-white pixels, and a distinct-colour count, then flags a frame that is essentially
  uniform. A black grab (minimized window, failed renderer, occluded swapchain) is otherwise
  indistinguishable from a legitimately dark scene.
  Exit code `2` means "grabbed, but it looks empty" and `3` means "no matching window", so a
  caller can branch on it without parsing text.
* **`--grid`** — overlays a coordinate grid with pixel labels. This is what makes the output
  usable for locating a UI element to click: read the coordinate off the annotated image
  instead of guessing.
* **`--mark` / `--annotate`** — record where something was clicked or what a shot is for,
  so a directory of shots stays self-describing.
* **`--frame client|window|extended`** — `client` (default) is the area the app renders into;
  `window` includes the title bar; `extended` uses the DWM extended frame bounds, which is
  the visually accurate rect for a borderless window.

## Notes

* Capture is a desktop `BitBlt` of the target rect, not `PrintWindow`. Minecraft runs
  borderless-windowed here, so the desktop grab is faithful; `PrintWindow` against a Vulkan
  swapchain generally returns black and is deliberately not offered. The flip side: an
  occluded window will be captured as occluded, so use `--focus` when it matters.
* Output is always UTF-8 even when piped. Window titles here are frequently CJK, and a piped
  stdout would otherwise be encoded with the ANSI code page and arrive mangled.
* Needs `Pillow`.
* DPI awareness is set per-monitor-v2, and the reported `dpi_awareness` field says which mode
  was actually obtained, so rects are in physical pixels unless that field says otherwise.
* Nothing here is Caustica-specific; it is generic and can be lifted elsewhere.
