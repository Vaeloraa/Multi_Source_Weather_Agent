# CasCast集合预测与天气不确定性数据

## 1. 压缩包内容与用途

这份数据用于后续的天气预测不确定性分析。它使用CasCast模型，对10个SEVIR测试集样本分别生成10种可能的未来降水演变，并计算10个集合成员的逐像素平均结果。

下游同学可以基于这些结果开展：

- 逐像素不确定性分析，例如集合标准差、分位数和超过指定强度阈值的概率；
- 风暴对象级不确定性分析，例如不同集合成员中强降水区域的位置、范围和移动差异；
- 直接预测分析，将集合平均作为中心预测，将成员之间的差异作为预测不确定性证据。

数据保持CasCast在SEVIR任务上的原生设置：13帧历史观测输入，12帧未来输出，空间分辨率为384×384。

## 2. 数据来源和样本编号

数据来自SEVIR测试集的雷达衍生垂直积分液态水产品，像素值采用SEVIR编码强度。

本次从测试集12159个样本中选取了以下10个样本：

| HDF5中的样本位置 | SEVIR测试集样本编号 |
|---:|---:|
| 0 | 7588 |
| 1 | 8873 |
| 2 | 52 |
| 3 | 4206 |
| 4 | 8573 |
| 5 | 7818 |
| 6 | 2721 |
| 7 | 700 |
| 8 | 1356 |
| 9 | 5455 |

样本筛选使用每个样本最后一帧历史观测。先筛选具有明显降水活动的候选样本，再根据降水强度、覆盖范围和空间分布特征进行确定性的最远点采样，使10个样本具有不同的降水形态。样本7588作为已完成样本保留在本次数据中。

## 3. 集合预测生成方式

每个样本按照以下流程生成：

1. 读取连续13帧历史观测；
2. 使用预训练EarthFormer确定性模型生成12帧未来预测；
3. 使用CasCast中的变分自动编码器将确定性预测编码为固定条件；
4. CasCast扩散模型使用10个固定随机种子，生成10个独立的12帧集合预测；
5. 在集合成员轴上计算逐像素算术平均，得到集合平均预测。

10个集合成员使用的随机种子依次为：

```text
2026080400, 2026080401, 2026080402, 2026080403, 2026080404,
2026080405, 2026080406, 2026080407, 2026080408, 2026080409
```

扩散推理步数为20，引导权重为2.0，条件潜变量随机种子为2026080399。同一个样本的10个集合成员共享同一份历史观测和确定性条件，集合成员之间的变化来自扩散生成过程的不同随机种子。

## 4. 时间含义

SEVIR序列相邻帧间隔5分钟。以最后一帧历史观测为0分钟：

- 13帧历史观测对应：-60、-55、-50、-45、-40、-35、-30、-25、-20、-15、-10、-5、0分钟；
- 12帧未来预测对应：+5、+10、+15、+20、+25、+30、+35、+40、+45、+50、+55、+60分钟。

## 5. 文件结构

```text
cascast_ensemble_uncertainty/
├── cascast_ensemble_10samples_384.h5
├── metadata.json
└── README.md
```

- `cascast_ensemble_10samples_384.h5`：全部像素数据；
- `metadata.json`：样本编号、数组形状、轴顺序、时间和推理参数；
- `README.md`：任务背景、生成方法和读取说明。

## 6. `metadata.json`字段说明

`metadata.json`是整个数据包的机器可读说明文件。程序可以先读取它，再根据其中记录的形状、轴顺序和时间信息解释HDF5数组。

### 6.1 顶层字段

| 字段 | 当前值或类型 | 含义 |
|---|---|---|
| `format_version` | `1` | 当前数据包的元数据格式版本 |
| `model` | `CasCast` | 生成集合预测所使用的模型 |
| `source_dataset` | `SEVIR VIL test split` | 输入样本来自SEVIR垂直积分液态水测试集 |
| `sample_indices` | 长度为10的整数列表 | 10个样本在SEVIR测试集中的原始编号；列表顺序与HDF5的样本轴顺序一致 |
| `sample_selection` | JSON对象 | 本次10个样本的筛选依据和筛选方法 |
| `dimensions` | JSON对象 | 样本数、集合成员数、时间帧数和空间尺寸 |
| `time` | JSON对象 | 相邻帧时间间隔，以及历史帧和预测帧相对0分钟的时间位置 |
| `ensemble` | JSON对象 | 集合成员随机种子及集合平均的定义 |
| `inference` | JSON对象 | CasCast扩散推理使用的参数 |
| `datasets` | JSON对象 | HDF5中每个数据集的形状、数据类型、轴顺序和内容 |

### 6.2 `sample_selection`：样本筛选信息

| 子字段 | 含义 |
|---|---|
| `source` | 指定筛选样本时使用的数据。本次使用每个样本最后一帧历史观测 |
| `method` | 先筛选有明显降水活动的样本，再按照降水特征进行确定性的最远点采样，以增加样本之间的差异 |

### 6.3 `dimensions`：整体数据尺寸

| 子字段 | 当前值 | 含义 |
|---|---:|---|
| `samples` | 10 | SEVIR样本数量 |
| `members_per_sample` | 10 | 每个样本生成的CasCast集合成员数量 |
| `history_frames` | 13 | 每个样本的历史观测帧数 |
| `forecast_frames` | 12 | 每个集合成员的未来预测帧数 |
| `height` | 384 | 每帧的纵向像素数 |
| `width` | 384 | 每帧的横向像素数 |

### 6.4 `time`：时间轴信息

| 子字段 | 含义 |
|---|---|
| `frame_interval_minutes` | 相邻原始帧之间相差5分钟 |
| `history_minutes` | 13帧历史观测相对于最后一帧历史观测的分钟数，从-60到0 |
| `forecast_minutes` | 12帧预测相对于最后一帧历史观测的分钟数，从+5到+60 |

例如，`forecast_minutes[6]`为35，表示HDF5预测时间轴下标6对应未来第35分钟。

### 6.5 `ensemble`：集合成员信息

| 子字段 | 含义 |
|---|---|
| `member_seeds` | 10个扩散随机种子，顺序与HDF5的集合成员轴一致 |
| `mean` | 集合平均的计算方法，即沿集合成员轴计算逐像素算术平均 |

例如，`member_seeds[4]`对应HDF5中`ensemble_members[:, 4]`使用的随机种子。

### 6.6 `inference`：推理参数

| 子字段 | 当前值 | 含义 |
|---|---:|---|
| `diffusion_steps` | 20 | CasCast扩散采样的迭代步数 |
| `guidance_weight` | 2.0 | 条件引导权重 |
| `condition_seed` | 2026080399 | 生成固定条件潜变量时使用的随机种子 |

### 6.7 `datasets`：HDF5字段目录

`datasets`下面的键与HDF5数据集名称一一对应，包括`sample_index`、`member_seed`、`history`、`ensemble_members`和`ensemble_mean`。每个数据集说明可能包含：

| 子字段 | 含义 |
|---|---|
| `shape` | 数组每一维的长度 |
| `dtype` | 数组在HDF5中的数据类型 |
| `axes` | 每一维依次代表的物理含义 |
| `content` | 数组保存的内容 |

`axes`中使用的轴名称含义如下：

| 轴名称 | 含义 |
|---|---|
| `sample` | 10个SEVIR样本 |
| `member` | 每个样本的10个集合成员 |
| `history_time` | 13个历史观测时刻 |
| `forecast_time` | 12个未来预测时刻 |
| `y` | 图像纵向像素坐标 |
| `x` | 图像横向像素坐标 |

读取元数据示例：

```python
import json

with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

sample_indices = metadata["sample_indices"]
forecast_minutes = metadata["time"]["forecast_minutes"]
members_shape = metadata["datasets"]["ensemble_members"]["shape"]
members_axes = metadata["datasets"]["ensemble_members"]["axes"]

print(sample_indices)
print(forecast_minutes)
print(members_shape)  # [10, 10, 12, 384, 384]
print(members_axes)   # ["sample", "member", "forecast_time", "y", "x"]
```

## 7. HDF5数据集

| 名称 | 数据类型 | 形状 | 轴顺序 | 内容 |
|---|---|---:|---|---|
| `sample_index` | 64位整数 | 10 | 样本 | SEVIR测试集样本编号 |
| `member_seed` | 64位整数 | 10 | 集合成员 | 每个集合成员对应的扩散随机种子 |
| `history` | 8位无符号整数 | 10×13×384×384 | 样本、历史时间、纵向像素、横向像素 | 13帧历史观测 |
| `ensemble_members` | 32位浮点数 | 10×10×12×384×384 | 样本、集合成员、预测时间、纵向像素、横向像素 | 每个样本的10个CasCast集合预测 |
| `ensemble_mean` | 32位浮点数 | 10×12×384×384 | 样本、预测时间、纵向像素、横向像素 | 10个集合成员的逐像素算术平均 |

例如，`ensemble_members[2, 4, 6]`表示：SEVIR测试集样本52的第5个集合成员，在第7个预测时刻，也就是预测起点后35分钟的384×384像素场。

## 8. Python读取示例

读取全部字段：

```python
import h5py

with h5py.File("cascast_ensemble_10samples_384.h5", "r") as f:
    sample_index = f["sample_index"][:]
    member_seed = f["member_seed"][:]
    history = f["history"][:]
    ensemble_members = f["ensemble_members"][:]
    ensemble_mean = f["ensemble_mean"][:]

print(sample_index)
print(history.shape)           # (10, 13, 384, 384)
print(ensemble_members.shape)  # (10, 10, 12, 384, 384)
print(ensemble_mean.shape)     # (10, 12, 384, 384)
```

读取一个样本并计算逐像素集合标准差：

```python
import h5py
import numpy as np

sample_position = 0

with h5py.File("cascast_ensemble_10samples_384.h5", "r") as f:
    sevir_sample_index = int(f["sample_index"][sample_position])
    members = f["ensemble_members"][sample_position]  # (10, 12, 384, 384)
    mean_prediction = f["ensemble_mean"][sample_position]

pixelwise_standard_deviation = members.std(axis=0, ddof=1)

print(sevir_sample_index)
print(mean_prediction.shape)               # (12, 384, 384)
print(pixelwise_standard_deviation.shape)  # (12, 384, 384)
```

计算像素超过指定强度阈值的集合概率：

```python
threshold = 133
exceedance_probability = (members >= threshold).mean(axis=0)
```

此时`exceedance_probability[t, y, x]`表示在第`t`个预测时刻，10个集合成员中该像素超过指定强度阈值的比例。
