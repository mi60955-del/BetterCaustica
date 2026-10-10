# cloud_preview.py — 平面云场离屏预览

把 `shaders/pipelines/world/volume.slang` 的云 coverage 场（hash / value noise / FBM / 风向框架
拉伸 / domain warp / 中尺度与侧风侵蚀 / 宏门控 / 云量 remap）逐行移植到 numpy，渲染成：

- `sheet_persp.png` — 从云平面下方仰视的透视视图（几何照抄 `planarCloudSkyInScatter`：
  `t = (planeY - camY)/d.y`、`tau = cov·dens·0.22·thick / max(|d.y|,0.04)`、`opacity = 1-e^-tau`、
  `color = cloudL + T·sky`；光照是替身，不是真 LUT）；
- `sheet_plan.png` — 云场俯视图（6 × 3.75 km 窗口，直接读条纹/云块/成片结构）。

用途：调云形态时在几秒内出对照图，不必每次进游戏。**这不是像素级复刻**——只复刻场的几何。

## 运行

```bash
python tools/cloud_preview.py
# 输出到 tools/cloud-preview-out/
```

依赖 numpy + PIL（`pip install numpy pillow`，或仓库自带的 uv 环境）。

## 约定

- `BASE` 字典必须与 `volume.slang` 的出厂常量保持一致，否则预览会骗人；每次改 shader 常量后同步这里。
- `SHEET_PERSP` / `SHEET_PLAN` 是要对比的变体列表，`variant(**kw)` 在 BASE 上做覆盖。
- 额外试验旋钮（`dir_jitter` / `cross_lo,hi` / `warp2` / `octaves` / `thresh_k` / `width_base,k`）
  可以先在预览里验证再决定是否落进 shader——v22 的那一轮就是这么定下来的。
