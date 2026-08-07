# CasCast × SEVIR 10事件评价摘要

- 数据：10个SEVIR VIL测试事件，每个事件13帧历史、10路CasCast ensemble、12帧预测和12帧ground truth。
- 主阈值：VIL ≥ 133。
- CasCast ensemble mean：CSI 0.498555，POD 0.621622，FAR 0.284233，RMSE 16.697，MAE 7.867。
- Persistence：CSI 0.333487，POD 0.501043，FAR 0.500694，RMSE 26.5521，MAE 12.6519。
- Ensemble验证：pixel std–absolute error Pearson 0.585496，80%中心区间实际覆盖率 0.491213，spread-skill ratio 0.276141。
- Storm object：CSI 0.26285，POD 0.331419，FAR 0.440443，匹配对象中心平均误差 8.171像素。

## 解释

CasCast在这10个事件上明显优于Persistence。标准差与真实误差呈正相关，说明ensemble分歧包含有用误差信息；但80%区间仅覆盖约49.1%的有效像素，且spread-skill ratio明显小于1，表明ensemble偏窄、离散度不足。以上结果只代表当前10个事件，不是完整SEVIR测试集论文复现。
