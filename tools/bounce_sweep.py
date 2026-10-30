#!/usr/bin/env python3
"""bounce_sweep.py — 弹射成本曲线的离线分析。

输入两份东西：

* ``rt-frame-stats/gpu.csv``（``-Dcaustica.rt.gpuTiming=true`` 时 ``RtGpuTiming``
  每帧写一行，列序 ``frame,skyLutMs,tracePrimaryMs,traceIndirectMs,dlssMs,
  exposureMs,bloomMs,lensMs,displayMs,copyMs``；文件每次启动清空，行内 ``frame``
  是会话内单调递增的帧号，丢帧时号码有洞）。
* 一个 arms JSON：本会话内逐臂记录的采样段。每臂一段
  ``frame_first..frame_last``（含端点，按帧号过滤而不是按行号），加上弹射数和
  角色（``sweep`` 扫描臂 / ``ref`` 穿插基准臂），可附 ``mc_perf`` 快照字段。

产出：每臂的每列中位数/p95，弹射逐级加一的边际毫秒（只对 ``sweep`` 臂，同场景
内按弹射数排序），以及基准臂相对扫描臂 4 档的漂移（会话内稳定度）。结果写
JSON，表格打 stdout。分析是离线的——它不碰游戏。

用法::

    python tools/bounce_sweep.py --gpu gpu.csv --arms arms.json [--out curve.json]

arms JSON 的形状::

    {
      "scene": "opaque-diffuse",
      "gpu_csv": "E:/Minecraft/26.2/rt-frame-stats/gpu.csv",
      "jar": "sha256:...",
      "camera": "(0,151,0) yaw 90 pitch 10, /time set 6000",
      "config": {"spp": 1, "ris-candidates": 0, "fog": false},
      "arms": [
        {"label": "B2", "bounces": 2, "role": "sweep",
         "frame_first": 120, "frame_last": 742,
         "fps": 61.2, "fps_1pct_low": 48.0, "frame_ms_p95": 17.3, "limit_reason": "NONE"}
      ]
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# gpu.csv 的表头第一列是帧号，其余列都是该帧的 pass 毫秒；名字由 RtGpuTiming.NAMES
# 生成，这里只要求「帧号 + 若干 *Ms 列」，新增 pass 列自动被带上。
FOCUS_COLUMNS = ("tracePrimaryMs", "traceIndirectMs")


def read_gpu_csv(path):
    """解析 gpu.csv，返回 (列名列表, {frame -> 行值列表})。

    表头行是 ``frame,<name1>Ms,...``。数据行帧号必须唯一；列数与表头不符的行
    跳过（写表与读表之间列数不会变，畸形行只可能是截断）。
    """
    header = None
    rows = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for record in csv.reader(fh):
            if not record:
                continue
            if header is None:
                if record[0] != "frame":
                    raise ValueError(f"{path}: first row is not a header: {record!r}")
                header = record
                continue
            if len(record) != len(header):
                continue
            try:
                frame = int(record[0])
                values = [float(v) for v in record[1:]]
            except ValueError:
                continue
            if frame in rows:
                raise ValueError(f"{path}: duplicate frame {frame}")
            rows[frame] = values
    if header is None:
        raise ValueError(f"{path}: no header row")
    return header, rows


def percentile(sorted_values, q):
    """线性插值百分位，与 numpy 默认一致；空列表是调用方的问题，这里直接崩。"""
    if not sorted_values:
        raise ValueError("percentile of empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def arm_stats(header, rows, arm, skip=0):
    """一臂的统计。帧号落在 [frame_first+skip, frame_last] 的行参与统计。

    ``skip`` 丢掉段头的过渡帧（改设置后瞬态），按帧号而不是行偏移丢。
    """
    columns = header[1:]
    first = arm["frame_first"] + skip
    last = arm["frame_last"]
    samples = [values for frame, values in rows.items() if first <= frame <= last]
    if not samples:
        return None
    stats = {"label": arm["label"], "bounces": arm["bounces"], "role": arm.get("role", "sweep"),
             "frames": len(samples)}
    for i, name in enumerate(columns):
        col = sorted(s[i] for s in samples)
        entry = {
            "median": percentile(col, 0.5),
            "p95": percentile(col, 0.95),
            "max": col[-1],
        }
        if name in FOCUS_COLUMNS or name == "total":
            entry["mean"] = sum(col) / len(col)
        stats[name] = entry
    total = sorted(sum(s) for s in samples)
    stats["total"] = {
        "median": percentile(total, 0.5),
        "p95": percentile(total, 0.95),
        "max": total[-1],
        "mean": sum(total) / len(total),
    }
    # mc_perf 快照原样带出，供报告引用；缺字段不报错。
    for key in ("fps", "fps_1pct_low", "fps_0p1pct_low", "frame_ms_p95", "frame_ms_max",
                "limit_reason"):
        if key in arm:
            stats[key] = arm[key]
    return stats


def marginal_costs(stats_by_bounces):
    """相邻弹射档之间的边际毫秒：Δmedian(traceIndirect)、Δmedian(tracePrimary)、Δtotal。

    ``stats_by_bounces``：弹射数 -> 该档的 arm_stats（只喂 sweep 臂）。
    """
    out = []
    levels = sorted(b for b in stats_by_bounces if stats_by_bounces[b] is not None)
    # 只有相邻档位（+1）才有边际可言；档位缺口必须断链，否则一档缺失会把
    # 多级的成本 smear 进一个"边际"数字里。
    for prev, cur in zip(levels, levels[1:]):
        if cur != prev + 1:
            continue
        a, b = stats_by_bounces[prev], stats_by_bounces[cur]
        step = {"from": prev, "to": cur}
        for key, source in (("d_traceIndirectMs", "traceIndirectMs"),
                            ("d_tracePrimaryMs", "tracePrimaryMs"),
                            ("d_total", "total")):
            step[key] = round(b[source]["median"] - a[source]["median"], 4)
        out.append(step)
    return out


def reference_drift(ref_stats, sweep_stat):
    """基准臂中位数与同弹射数扫描臂中位数的最大偏离（毫秒，取两列关注列）。

    会话内噪声底：偏离明显大于它说明会话漂移，边际数字不可信。
    """
    if ref_stats is None or sweep_stat is None:
        return None
    return {
        name: round(abs(ref_stats[name]["median"] - sweep_stat[name]["median"]), 4)
        for name in FOCUS_COLUMNS
    }


def analyse(arms_doc, skip=0):
    gpu_path = arms_doc.get("gpu_csv")
    header, rows = read_gpu_csv(gpu_path)
    scenes = {}
    for arm in arms_doc["arms"]:
        scenes.setdefault(arm["scene"], []).append(arm)

    report = {"gpu_csv": gpu_path, "columns": header[1:], "scenes": {}}
    for scene, arms in scenes.items():
        per_arm = [arm_stats(header, rows, arm, skip=skip) for arm in arms]
        sweep = {a["bounces"]: st for a, st in zip(arms, per_arm)
                 if st is not None and st["role"] == "sweep"}
        refs = [st for a, st in zip(arms, per_arm)
                if st is not None and st["role"] == "ref"]
        # 穿插基准所在的档位（协议里是默认 4）没有自己的 sweep 臂；该档的代表性
        # 数字取全部 ref 臂的中位数均值，否则边际链会在那里断开。
        chain = dict(sweep)
        by_bounces = {}
        for st in refs:
            by_bounces.setdefault(st["bounces"], []).append(st)
        for bounces, sts in by_bounces.items():
            if bounces in chain:
                continue
            rep = json.loads(json.dumps(sts[0]))
            rep["role"] = "ref-representative"
            rep["ref_arm_count"] = len(sts)
            for source in ("tracePrimaryMs", "traceIndirectMs", "total"):
                for stat in ("median", "p95", "mean", "max"):
                    if stat in rep[source]:
                        rep[source][stat] = round(
                            sum(s[source][stat] for s in sts) / len(sts), 4)
            chain[bounces] = rep
        sweep_stat_4 = chain.get(4)
        for st in refs:
            st["drift_vs_sweep4"] = reference_drift(st, sweep_stat_4)
        report["scenes"][scene] = {
            "arms": [st for st in per_arm if st is not None],
            "marginal": marginal_costs(chain),
            "missing_bounces": sorted(
                set(arm["bounces"] for arm in arms if arm["role"] == "sweep")
                - set(sweep)),
        }
    return report


def format_table(report):
    lines = []
    for scene, data in report["scenes"].items():
        lines.append(f"== {scene} ==")
        cols = [c for c in report["columns"] if c in FOCUS_COLUMNS] + ["total"]
        head = (f"{'arm':>6} {'B':>2} {'role':>5} {'n':>5} "
                + " ".join(f"{c+' med/p95':>22}" for c in cols))
        lines.append(head)
        for st in data["arms"]:
            cells = " ".join(f"{st[c]['median']:9.3f}/{st[c]['p95']:10.3f}" for c in cols)
            lines.append(f"{st['label']:>6} {st['bounces']:>2} {st['role']:>5} {st['frames']:>5} "
                         + cells)
        for step in data["marginal"]:
            lines.append(f"  marginal {step['from']}->{step['to']}: "
                         f"indirect {step['d_traceIndirectMs']:+.3f} ms, "
                         f"primary {step['d_tracePrimaryMs']:+.3f} ms, "
                         f"total {step['d_total']:+.3f} ms")
        for st in data["arms"]:
            if st.get("drift_vs_sweep4"):
                d = st["drift_vs_sweep4"]
                lines.append(f"  ref drift {st['label']}: "
                             + ", ".join(f"{k} {v:.3f} ms" for k, v in d.items()))
        if data["missing_bounces"]:
            lines.append(f"  MISSING sweep arms for bounces {data['missing_bounces']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", help="rt-frame-stats/gpu.csv；arms 里给了 gpu_csv 时可省")
    parser.add_argument("--arms", required=True, help="会话内逐臂记录的 JSON")
    parser.add_argument("--out", help="结果 JSON 输出路径")
    parser.add_argument("--skip", type=int, default=0,
                        help="每臂段头丢弃的过渡帧数（按帧号）")
    args = parser.parse_args(argv)

    with open(args.arms, encoding="utf-8") as fh:
        arms_doc = json.load(fh)
    if args.gpu:
        arms_doc["gpu_csv"] = args.gpu
    if not arms_doc.get("gpu_csv"):
        parser.error("no gpu.csv: pass --gpu or set gpu_csv in the arms JSON")

    report = analyse(arms_doc, skip=args.skip)
    text = format_table(report)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
