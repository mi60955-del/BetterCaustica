# bake_cloud_vols.py — 体积云密度体积烘焙

程序化生成 `RtCloudVolumes.java` 光追管线吃的九个云密度体积（`textures/cloud/*.vol`，
clouds_unreal / Ryan Brucks 方案的烘焙体积），带布局校验和投影对比图。替换或新调云形状时
用几秒到一分钟的离线烘焙代替进游戏反复看。

## 运行

```bash
python tools/bake_cloud_vols.py                 # 烘焙九个形状，写入 assets，校验，出对比图
python tools/bake_cloud_vols.py --check-only    # 只校验已写盘的文件（回读 + 全部门禁）
python tools/bake_cloud_vols.py --sheet out.png # 自定义对比图输出路径
```

依赖 numpy + PIL。校验有 fail 时退出码非 0，可以直接当 CI/钩子的门。

## 二进制契约（与 RtCloudVolumes.java 一一对应）

- 8 字节头：`"BVOL"` + u8 version(1) + u8 保留 + u16 小端分辨率(128)；
- 之后 128³ 字节 R8 密度，顺序 x 最快、其次 y、**其次 z**；
- **z 轴是竖直方向**：Java 侧把每个 z 切片上传成一个 2D 数组 layer，
  `volume.slang` 的 `cloudInstanceDensity` 把 layer 索引当高度采，layer 0 = 云底；
  层内 (x, y) 是实例盒的水平足迹。

## 纵横比预补偿（容易踩的坑）

`volume.slang` 的实例盒把体积拉伸为 `cloudSpan`（footprint）: `cloudBoxHeight`（高）=
12 : 2.4 = **恒定 5:1**。所以脚本在 1.0 × 1.0 × `VERTICAL_SCALE(0.2)` 的世界比例域里摆
puff 几何再采样进 128³ 网格——直接按立方体直觉烘焙，进游戏所有云都会被压成饼。

## 校验门（针对 0.2.3 坏资产的失败模式）

- 最外层体素必须为 0（clamp-to-edge 采样读的就是它）；`BORDER_VOXELS`(4) 内是渐变带；
- 任一轴的切片均值不得恒定（体积被拉伸）或出现相邻切片悬崖（条纹特征）；
- 竖直剖面必须底重顶轻；填充率与均值密度限幅。

对比图 `bake_check_sheet.png` 每行一个形状：max x / max y / 俯视 / 40% 高度切片——
**三个方向的轮廓都必须像云**，这是形状验收的最终标准。

## 调形状

- `_shape_puffs()`：puff 布局（footprint、数量、包络收缩、竖直扁平度）。puff 高度是
  分层随机撒的——离散"层"会在侧视投影里读出竖直条纹，别改回去；
- `FINISH`：每形状的侵蚀/细节/对比度旋钮，对着对比图调；
- 每个形状的种子取自文件名，确定性可复现；
- 形状顺序必须与 `RtCloudVolumes.RESOURCES` 完全一致（游戏 Cloud Type 滑条就是这张表），
  每次运行都会交叉核对，不一致直接报错退出。

改完 `.vol` 需要重启游戏（RtCloudVolumes 只在 RT 初始化时读一次，F3+T 不重载）。
旧的 0.2.3 坏资产备份在 `cloud-preview-out/replaced_vols/`。
