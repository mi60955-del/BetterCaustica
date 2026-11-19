# cloud_preview.py — 程序化云场离屏预览

把 `RtCloudNoise.generate`（512² 可平铺噪声图集）和 `volume.slang` 的**低积云层**形状场
逐行移植到 numpy：带偏置的覆盖阈值、平底/圆顶剖面、四次幂侵蚀、风切变。渲染成三张对照图：

- `sheet_persp.png` — 从 y=64 仰视的透视视图。几何照抄 `cloudPlanarStack` 的穿越
  （`t = (mid - camY)/d.y`、`tau = density·0.5·tallness / max(|d.y|,0.04)`、
  `opacity = 1-e^-tau`、`color = cloud + T·sky`）。光照是天空梯度替身，不是真 LUT。
- `sheet_plan.png` — 云场俯视图（6 × 4 km 窗口），用来看云块、间隙和成片结构。
- `sheet_cross.png` — 垂直剖面，用来看平底、圆顶和有没有横向条纹。

用途：调云形态常量时几秒出对照图，不必每次进游戏。**不是像素级复刻**：Python 的舍入不同、
光照是替身、只画低积云层；忠实复刻的是**场的几何**。

## 运行

```bash
./.venv/Scripts/python.exe tools/cloud_preview.py
# 输出到 tools/cloud-preview-out/
```

依赖 numpy + PIL（仓库的 `.venv` 里有）。

## 约定

- 顶部常量必须与 `volume.slang` 的 `CLOUD_*` 和 `RtCloudNoise` 的 `PERIODS/SEEDS/GAIN` 一致，
  否则预览会骗人；每次改 shader 常量后同步这里。
- `VARIANTS` 是要对比的变体列表（覆盖阈值 / 斑块度），改它就能在预览里比较候选值，
  验证过再落进 shader。
- 高积云、雨层云、卷云没有画进预览：它们共用同一个阈值，只是层表名不同。
