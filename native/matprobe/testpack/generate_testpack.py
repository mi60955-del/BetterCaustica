"""Generate a synthetic LabPBR resource pack for exercising matprobe.

The pack is not meant to look like Minecraft. It is meant to make every branch of the
LabPBR decode reachable by eye and by number:

  * a dielectric whose F0 comes from the green byte
  * near-mirror and fully-rough smoothness extremes
  * a conductor enum byte (gold, 231) and an authored-F0 conductor (255)
  * emission through the alpha byte
  * the blue byte's two mutually exclusive lanes, porosity (<= 64) and scattering (> 64)
  * a _n map carrying tangent normals, ambient occlusion and a non-flush height field

Run it with the interpreter that has Pillow, then point matprobe at ./pack:

    python generate_testpack.py
    matprobe.exe --pack testpack/pack --material matprobe_test:block/matte_stone --out out
"""

import math
import os
from PIL import Image

SIZE = 16
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pack")
TEXTURES = os.path.join(ROOT, "assets", "matprobe_test", "textures", "block")

# name, albedo base, albedo accent, smoothness, green byte, blue byte, alpha byte, has _n
MATERIALS = [
    ("matte_stone",   (0x7A, 0x76, 0x70), (0x5E, 0x5A, 0x55), 0.15, 10, 0, 255, False),
    ("mirror_steel",  (0xC8, 0xCC, 0xD2), (0x9A, 0x9E, 0xA4), 1.00, 10, 0, 255, False),
    ("gold_metal",    (0xD4, 0xA5, 0x3F), (0xB0, 0x86, 0x2C), 0.80, 231, 0, 255, False),
    ("tinted_metal",  (0x62, 0x8A, 0xB4), (0x46, 0x68, 0x8E), 0.70, 255, 0, 255, False),
    ("emissive_lamp", (0xE8, 0xE2, 0xC0), (0xD0, 0xC8, 0xA0), 0.30, 10, 0, 128, False),
    ("waxy_sss",      (0xC0, 0x96, 0x92), (0xA8, 0x7C, 0x78), 0.45, 10, 200, 255, False),
    ("wet_sand",      (0xC4, 0xB2, 0x84), (0xA8, 0x96, 0x6C), 0.20, 10, 64, 255, False),
    ("bumpy_wood",    (0x8A, 0x62, 0x3C), (0x6E, 0x4C, 0x2C), 0.35, 10, 0, 255, True),
]


def make_albedo(base, accent, period=4):
    image = Image.new("RGBA", (SIZE, SIZE))
    pixels = image.load()
    for y in range(SIZE):
        for x in range(SIZE):
            checker = ((x // period) + (y // period)) % 2 == 0
            rgb = base if checker else accent
            # A gentle ramp keeps neighbouring texels distinct, so a specular highlight or a
            # normal-map wobble has something to vary across.
            shade = 0.85 + 0.3 * (x / (SIZE - 1))
            pixels[x, y] = (int(rgb[0] * shade), int(rgb[1] * shade), int(rgb[2] * shade), 255)
    return image


def make_spec(smoothness, green, blue, alpha):
    return Image.new("RGBA", (SIZE, SIZE),
                     (round(smoothness * 255), green, blue, alpha))


def make_normal(amplitude=0.55):
    """Tangent normals from a sinusoidal height field, with AO and the matching height."""
    image = Image.new("RGBA", (SIZE, SIZE))
    pixels = image.load()
    for y in range(SIZE):
        for x in range(SIZE):
            dx = amplitude * math.cos(2.0 * math.pi * x / SIZE)
            dy = amplitude * math.cos(2.0 * math.pi * y / SIZE)
            nz = math.sqrt(max(0.0, 1.0 - dx * dx - dy * dy))
            # Height is a depth: 255 is flush with the flat surface, lower values sink in.
            height = 0.55 + 0.45 * nz
            pixels[x, y] = (int((dx * 0.5 + 0.5) * 255), int((dy * 0.5 + 0.5) * 255),
                            255, int(height * 255))
    return image


def main():
    os.makedirs(TEXTURES, exist_ok=True)
    for name, base, accent, smoothness, green, blue, alpha, has_normal in MATERIALS:
        make_albedo(base, accent).save(os.path.join(TEXTURES, name + ".png"))
        make_spec(smoothness, green, blue, alpha).save(os.path.join(TEXTURES, name + "_s.png"))
        if has_normal:
            make_normal().save(os.path.join(TEXTURES, name + "_n.png"))
    with open(os.path.join(ROOT, "pack.mcmeta"), "w", encoding="utf-8") as handle:
        handle.write('{\n  "pack": {\n    "pack_format": 15,\n'
                     '    "description": "matprobe synthetic LabPBR test pack"\n  }\n}\n')
    print("wrote", len(MATERIALS), "materials to", TEXTURES)


if __name__ == "__main__":
    main()
