# lens_preview.py — 镜头光晕离屏预览

把 `shaders/pipelines/lens/` 的三段算法逐行移植到 numpy，在一张合成 HDR 场景上出图：

- `source.comp.slang` 的源场：4×4 双线性采样做盒平均，按曝光后的亮度阈值取超额、软饱和；
- `main.comp.slang` 的星芒（沿光圈叶片方向的双向 gather）、光环（三通道不同半径的环采样）和
  镜头污渍（程序化灰尘场 × 画面亮度）；
- 合成到场景上，和游戏里 display pass 的加法位置一致（都在色调映射之前、以场景线性光相加）。

用途：调星芒长度、叶片数、光环半径与色散、灰尘密度时几秒出对照图，不必每次进游戏等世界加载。
**这不是像素级复刻**——它复刻的是几何与能量，纹理过滤、采样抖动以外的差异（例如实际 rtImage 的
噪声、DLSS-RR 的输出特性）不在范围内。

## 运行

```bash
python tools/lens_preview.py --width 1920 --height 1080
# 输出到 tools/lens-preview-out/1080/lens-sheet.png
```

`--divisor` 控制光晕图（flare image）相对画面的尺寸，对应游戏里的「Flare Resolution」（默认 8）；
源场（source field）固定在 1/8，不跟这个值变化，因为光线宽度等于源纹素宽度，源变细会把星芒变成扇形。
`--sample-rate` 对应「Flare Sampling」，按与 Java 相同的取整规则放大或缩小两处汇聚的采样数。
其余参数对应设置项：
`--starburst --spikes --spike-length --samples --halo --halo-radius --dispersion --dirt --dirt-scale --halo-taps`。

每次运行会打印源场与光晕图各自的尺寸、每个阶段每纹素的采样数和总取次数，以及折算到每个显示像素
的取次数；改采样数之前先看这一行，默认设置下每像素大约 4~5 次取纹，1/4 分辨率约 18 次。

## 约定

- `SHADER_*` 常量必须与 `source.comp.slang` / `main.comp.slang` 里的 `static const` 同名同值，
  否则预览会骗人。`tools/test_lens_preview.py` 会解析两边的声明并逐条比对，改动 shader 常量后跑：

  ```bash
  python -m unittest tools.test_lens_preview
  ```

  同一个测试还检查行为（阈值以下为零、软饱和单调有界、星芒为 2×叶片数条、光环峰在半径处、
  红环在外、蓝环在内、无光时灰尘为零、灰尘总量）和成本预算（每显示像素取纹数 < 5、
  光晕纹理面积为画面的 1/64），以及 `CausticaConfig` 的默认值和 `RtComposite` 里的 1/8 尺寸。
- 抖动（`texel_dither`）不是精度误差而是成像的一部分：1/8 纹理上的等距采样会在环和射线上留下
  串珠，按纹素相位错开后由双线性放大接上。删掉它预览会重新变成一串珠子。
- 星芒的每个采样必须双向取值：gather 只从它读的一侧带光回来，单向采样只能点亮射线的一半。

## 进游戏核对

`docs/developer_guide.md` 的 “Per-pass GPU timing” 一节：`-Dcaustica.rt.gpuTiming=true`
会把每段 GPU 时间写进 `rt-frame-stats/gpu.csv`，其中 `lensMs` 就是这两次 dispatch 的合计。基准：
1920×1080 输出、1/8 纹理、出厂设置下约 0.20 ms；`tools/lens-worst-case.sh` 会把采样数拉到上限
再测一遍（约 0.49 ms）。
