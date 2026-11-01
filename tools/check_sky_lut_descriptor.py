#!/usr/bin/env python3
"""Check the host sky-view descriptor stages against the shipped raygen SPIR-V."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtPipeline.java"
SHADER_DIR = (
    ROOT
    / "build/generated/shaders/bettercaustica/shaders/pipelines/world"
)
RAYGENS = ("primary.rgen.spv", "indirect.rgen.spv", "indirect_ser.rgen.spv")


def host_stages() -> str:
    source = PIPELINE.read_text(encoding="utf-8")
    start = source.index("binds.get(WORLD_SKY_VIEW)")
    end = source.index("binds.get(WORLD_TRANSMITTANCE)", start)
    block = source[start:end]
    stages = re.search(r"\.stageFlags\(([^;]+?)\);", block, re.DOTALL)
    if stages is None:
        raise RuntimeError("WORLD_SKY_VIEW has no descriptor stage flags")
    return " ".join(stages.group(1).split())


def raygen_sky_samples(path: Path) -> int:
    result = subprocess.run(
        ["spirv-dis", str(path), "--no-color"],
        check=True,
        capture_output=True,
        text=True,
    )
    assembly = result.stdout
    entry = next(line for line in assembly.splitlines() if "OpEntryPoint" in line)
    if "RayGenerationKHR" not in entry:
        raise RuntimeError(f"{path.name} is not a ray-generation shader")

    name = re.search(r'OpName\s+%([A-Za-z0-9_]+)\s+"skyViewLut"', assembly)
    if name is None:
        raise RuntimeError(f"{path.name} does not declare skyViewLut")
    variable = name.group(1)
    if not re.search(
        rf"OpDecorate\s+%{re.escape(variable)}\s+Binding\s+10\b", assembly
    ):
        raise RuntimeError(f"{path.name}: skyViewLut is not binding 10")
    if not re.search(
        rf"OpDecorate\s+%{re.escape(variable)}\s+DescriptorSet\s+0\b", assembly
    ):
        raise RuntimeError(f"{path.name}: skyViewLut is not descriptor set 0")
    return len(re.findall(rf"OpLoad\s+%[A-Za-z0-9_]+\s+%{re.escape(variable)}\b", assembly))


def main() -> int:
    stages = host_stages()
    print(f"WORLD_SKY_VIEW layout stages: {stages}")
    failures = []
    for filename in RAYGENS:
        path = SHADER_DIR / filename
        if not path.is_file():
            raise RuntimeError(f"missing {path}; run ./gradlew compileShaders first")
        samples = raygen_sky_samples(path)
        print(f"{filename}: RayGenerationKHR, skyViewLut set=0 binding=10, loads={samples}")
        if samples == 0:
            raise RuntimeError(f"{filename} declares but never samples skyViewLut")
        if "VK_SHADER_STAGE_RAYGEN_BIT_KHR" not in stages:
            failures.append(filename)

    if failures:
        print(
            "FAIL: these ray-generation shaders sample a descriptor whose layout "
            "does not enable VK_SHADER_STAGE_RAYGEN_BIT_KHR:"
        )
        for filename in failures:
            print(f"  - {filename}")
        print("Vulkan leaves this access undefined; a black sky fill is a valid symptom.")
        return 1

    print("PASS: every shipped raygen that samples skyViewLut is enabled in its descriptor layout.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
