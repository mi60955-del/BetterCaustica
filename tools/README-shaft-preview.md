# shaft_preview.py — 太阳光束（sun shafts）离屏预览与成本模型

把 `shaders/pipelines/world/volume.slang` 里的光束积分器（`shaftInScatter()`，:1293）与解析雾积分器
（`fogSegmentInScatter()`，:1127）逐行移植到 numpy，在同一张合成场景上出图并计数。

要验证的命题有两条：

1. **它算不算体积效果？** 光束项的形式是 `L = structure · strength · phase · E · Σ_k(σ·T_k·V_k)`，
   其中 `V_k` 是第 k 层采样点朝太阳的阴影射线给出的可见性，`structure = 1 − mean(V)`。
   介质是常数，光只被遮挡关系开合：把雾密度置零、只开光束，路径上依然有光；把可见性强制为 1
   （均匀受光），这一项必须**恰好为零**，因为门控 `structure` 成了零。一个体积效应在同样控制下仍会
   把路径点亮——这就是它与体积项的分界。
2. **它花多少？** 每个命中像素 8 根阴影射线 + 8 次云影场求值，是解析雾每像素 4 + 4 的两倍，是行军雾
   （`FOG_MARCH_PROBE_STRATA = 8`）的 1.0 倍；而整帧的像素成本与光束远近、密度、强度**全部无关**，
   只有 `shaftWeight ≤ 1e-4` 的误差阈值会让它整段不采样。

**这不是像素级复刻**：天空 LUT 用解析梯度替代，云层用两倍频替身，水的分支由测试数值覆盖而不是渲染。
它复刻的是「光从哪来」「花几次」，这两点正是命题的全部。

## 运行

```bash
python tools/shaft_preview.py                      # 审计 + 预算表 + 4×2 对照图
python tools/shaft_preview.py --budget-only        # 只做源码审计与预算，不出图
python tools/shaft_preview.py --width 960 --height 540 --frames 8
```

输出在 `tools/shaft-preview-out/`：`sheet.png` 是 4 列对照图（关 / 光束 / 控制 / 雾 / 两者 / 纯光束项 /
结构门控 / RR 反照率份额），另外每个臂各存一张 PNG。

参数对应设置项：`--reach`（`volume.shaft-distance`）、`--shaft-density`（`volume.shaft-density`）、
`--fog-density`（`volume.fog-density`）、`--frames`（抖动累加帧数）。采样次数与这些参数无关，只有命中
像素数会缩放总数。

## 读数

- **probe budget**：由 shader 常量推出的每调用预算，以及折算到整帧的射线数。光束一行是 8 根射线，
  解析雾 4 根、行军雾 8 根——「同样的消耗」这一句在数值上就是 2.00× / 1.00×。
- **effect**：雾密度为零时光束臂的均值/峰值、控制臂（V≡1）的最大值（必须恰好 0）、结构门控的均值与
  变异系数。两个 PASS 是这一节的判据。
- **per-pixel constancy**：把 reach 与 density 拉到两端重渲，打印每个命中像素的射线数（恒 8.00）和该
  设置下实际的 in-scatter 均值（会差十几倍）。成本不动、效果在动，这就是「同样的消耗」。
- **gating**：源码审计，检查两个积分器的每个调用点是否都在 `CAUSTICA_VOLUME_PATH` 之下、空气/水下
  分支是否只受自己的开关控制、`leanVolumePath` 是否仍要求 `!shaftOn`，以及 `build.gradle` 是否仍编
  三份 `_lean` raygen。

## 约定

- `SHADER_*` 常量必须与 `world/volume.slang` 的 `static const` 同名同值，`DEFAULT_*` 必须与
  `CausticaConfig.java` 的 `Rt.Volume` 默认值一致，否则预览会骗人。`tools/test_shaft_preview.py`
  解析两边声明逐条比对，改动任何一侧之后跑：

  ```bash
  python -m unittest tools.test_shaft_preview
  ```

  同一个测试还检查行为（控制臂恰好为零、雾为零仍有光、全阴影空气不发亮、水下结构下限 0.45、
  等散射权重分层边界）和成本（每像素 8 根射线 + 8 次云影、解析雾 4 根、误差阈值以下不采样、
  同一帧里光束恰为雾的两倍），以及上面那几条 gating 断言。
- 分层边界按「等散射权重」而不是「等距离」：`shaftDepthAtWeight` 用 `−ln(1−k·total/8)/σ`。
- 本仓库的可见性来自硬件阴影射线（`visibility()`），不是阴影图采样。所以严格说它比「阴影图反推」
  更贵：每个采样点都是一次真实求交，而不是一次贴图采样——这正是这份预览要量出来的地方。
- 光束项**不参与** `primary.rgen.slang` 的 RR 反照率混合（:152 起），因为它靠遮挡结构产生高频，
  写进 albedo guide 会被重建当成材质、留下格纹。预览里的 `share` 面板就是那个份额，仅供对照。

## 进游戏核对

`docs/developer_guide.md` 的 “Per-pass GPU timing” 一节：`-Dcaustica.rt.gpuTiming=true` 会把每段
GPU 时间写进 `rt-frame-stats/gpu.csv`。光束与雾都在 `tracePrimary` / `traceIndirect` 里，没有独立的
计数器，所以对照实验要切开关而不是看某一列：同一机位下依次关闭 `volume.shafts` 与 `volume.fog`，
比较 `tracePrimaryMs + traceIndirectMs`。改动前后先确认日志里有

```
Using graphics backend Vulkan
DLSS-RR feature created: <render> -> <display>
```

否则测到的是 OpenGL 路径上根本不执行的光线追踪，数字会「好看」得没有意义。

### 实测结果

`RT测试` 存档里封了一个黑色混凝土房间（y 149–157，天花板开三个洞），`/time set 6000` 固定正午太阳，
机位 (0, 151, 0)、yaw 90 / pitch 10，1920×1080 显示、960×540 追踪、DLSS-RR preset 6。
`gpu.csv` 每次启动都会重写，所以每个臂各占一次会话；下表是静止机位最后 240 帧的均值。

| 臂 | `volume.shafts` | `volume.fog` | tracePrimary | traceIndirect | 画面 |
| --- | --- | --- | --- | --- | --- |
| 全关 | off | off | 0.451 ms | 6.532 ms | lean raygen |
| 只开光束 | on | off | 1.994 ms | 16.866 ms | 有光束 |
| 只开雾 | off | on | 1.699 ms | 16.030 ms | 均匀雾 |
| 光束密度 0 | on (density 0) | off | 1.720 ms | 14.572 ms | 与全关一致 |

相对全关：只开光束 +1.54 / +10.33 ms，只开雾 +1.25 / +9.50 ms，而**密度为 0 的光束**（画面与全关
完全一致、积分器在 `sigma ≤ 0` 处直接返回）已经要 +1.27 / +8.04 ms。也就是说大头不是那 8 根射线
（光束臂减雾臂只多 0.30 / 0.84 ms），而是「打开光束开关 → 编译后的体积路径 raygen 被选中」这件事：
`leanVolumePath` 里带着 `!shaftOn`，所以开光束必然放弃 `_lean` 孪生、连带把雾/体积代码的寄存器成本
一起背上（`./gradlew` 构建时 `spirv-dis` 的指令数：`primary.rgen.spv` 51252 条 vs
`primary_lean.rgen.spv` 13179 条，`indirect` 64949 vs 26368）。

截图在 `tools/shaft-preview-out/ingame/`（该目录被 .gitignore 忽略）。
