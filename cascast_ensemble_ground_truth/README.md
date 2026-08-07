# CasCast集合预测对应的真实未来

本目录保存`cascast_ensemble_uncertainty`中10个SEVIR样本对应的真实未来，用于预测结果评价。样本编号、样本顺序、时间范围和空间尺寸均与集合预测文件一致。

## 文件

- `cascast_ground_truth_10samples_384.h5`：10个样本的真实未来像素数据；
- `metadata.json`：样本编号、数组形状、轴顺序和预测时间。

## HDF5数据集

| 名称 | 数据类型 | 形状 | 轴顺序 | 内容 |
|---|---|---:|---|---|
| `sample_index` | 64位整数 | 10 | 样本 | SEVIR测试集样本编号 |
| `ground_truth` | 8位无符号整数 | 10×12×384×384 | 样本、预测时间、纵向像素、横向像素 | 12帧真实未来 |

12帧真实未来分别对应预测起点后的+5、+10、+15、+20、+25、+30、+35、+40、+45、+50、+55和+60分钟。

## 与集合预测对应读取

```python
import h5py
import numpy as np

with h5py.File(
    "../cascast_ensemble_uncertainty/cascast_ensemble_10samples_384.h5", "r"
) as prediction_file, h5py.File(
    "cascast_ground_truth_10samples_384.h5", "r"
) as truth_file:
    prediction_sample_index = prediction_file["sample_index"][:]
    truth_sample_index = truth_file["sample_index"][:]
    assert np.array_equal(prediction_sample_index, truth_sample_index)

    ensemble_members = prediction_file["ensemble_members"][:]
    ensemble_mean = prediction_file["ensemble_mean"][:]
    ground_truth = truth_file["ground_truth"][:]

print(ensemble_members.shape)  # (10, 10, 12, 384, 384)
print(ensemble_mean.shape)     # (10, 12, 384, 384)
print(ground_truth.shape)      # (10, 12, 384, 384)
```
