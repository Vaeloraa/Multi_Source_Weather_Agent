# 任务设计

 [question_template_matching.csv](D:/Profile/专业实习/Zephyrus/scripts/question_template_matching)

## TASK1 基础事件预测：请分析该事件未来60分钟的VIL变化，并说明不同时效的预测结果。

### 1.指标

| 情况             | 预测     | 真实GT   | 含义             |
| ---------------- | -------- | -------- | ---------------- |
| Hit              | 强回波   | 强回波   | 正确预测         |
| Miss             | 非强回波 | 强回波   | 漏报             |
| False alarm      | 强回波   | 非强回波 | 虚警             |
| Correct negative | 非强回波 | 非强回波 | 正确预测普通区域 |

| 指标 | 数值  | 理解                            | **公式**                                     |
| ---- | ----- | ------------------------------- | -------------------------------------------- |
| CSI  | 0.682 | 预测与真实强回波区域重合程度    | CSI = Hit ÷ (Hit + Miss + False alarm)       |
| POD  | 0.833 | 真实强回波像素中被预测出来的    | POD = Hit ÷ (Hit + Miss)                     |
| FAR  | 0.209 | 预测为强回波的像素中的属于虚警  | FAR = False alarm ÷ (Hit + False alarm)      |
| RMSE | 9.388 | 对较大VIL误差惩罚更强，越低越好 | RMSE = √平均[(预测值 - 真实值)²]             |
| MAE  | 2.410 | 所有有效像素的平均绝对VIL误差   | MAE = 所有有效像素\|预测值 - 真实值 \|的平均 |

#### 2.data解读

1. ##### `CATALOG.csv`

这是SEVIR官方数据目录

目前Data Indexer和统计工具使用了：

```
id
file_name
file_index
img_type
time_utc
event_type
四个经纬度边界
data_min
data_max
pct_missing
```

目前还没有充分使用：

```
minute_offsets
episode_id
event_id
proj
size_x、size_y
height_m、width_m
```

2. ##### **`cascast_ensemble_cache`**

​	为一个CasCast测试样本建立的“事件清单信息”

  3.**`cascast_evaluation_report.json`**

这是利用10个事件预测和GT计算出的完整实验报告，包含：

- 六个VIL阈值下的CSI、POD、FAR；
- +5至+60分钟逐时效评价；
- CasCast与Persistence比较；
- 每个事件的指标；
- Ensemble区间覆盖率和可靠性；
- Rank histogram；
- Storm-object评价。

它属于**分析结果**，不是模型输入。

向老师解释：

> 在10个事件缓存建立后，我统一计算了多阈值、逐时效、基线对比和不确定性评价，完整机器可读结果保存在该JSON中。

------

##### 4. `evaluation_exports`

这是将完整JSON整理成便于查看和汇报的文件。

| 文件                          | 含义                               |
| ----------------------------- | ---------------------------------- |
| `summary.md`                  | 10事件实验摘要                     |
| `threshold_metrics.csv`       | 不同VIL阈值下的指标                |
| `lead_time_metrics.csv`       | +5～+60分钟逐时效指标              |
| `event_metrics.csv`           | 每个事件的CasCast和Persistence指标 |
| `ensemble_member_metrics.csv` | 10个成员分别的表现                 |
| `ensemble_reliability.csv`    | 预测概率与真实发生频率             |
| `dataset-evaluation-page.png` | 整体评价网页截图                   |

这些都是从`cascast_evaluation_report.json`导出的，不能被说成原始数据。

向老师解释：

> `evaluation_exports`是实验结果的展示层，便于画图、检查和汇报，不参与模型预测。

------

##### 5. `uploads`

目前只有`.gitkeep`，用于保留空目录。

当前用户上传的图片和13帧序列主要保存在服务器内存中：

- 不会永久保存到该目录；
- 服务器重启后上传内容消失；
- 这是一种Demo级隐私和简化设计。

不要向老师说上传数据已经持久化。

------

##### 6 `.gitkeep`

它没有数据含义，只是让Git能够保留空的`data`目录。



#### 3.为什么预测的比真实的模糊？

当前页面中的“预测原图”实际展示的是10路CasCast ensemble mean。不同成员的回波位置存在偏差，取平均会平滑边缘；同时CasCast在48×48 latent空间生成，再通过Autoencoder解码到384×384，也会损失部分小尺度结构。所以它比原始SEVIR ground truth模糊。下一步应该分别比较单成员、ensemble mean以及Autoencoder重建结果，区分平均平滑和压缩损失的贡献。



## TASK2 Hazard分析：未来60分钟哪些区域可能出现未来60分钟哪些区域可能出现较强回波？分别什么时候出现、持续多久？

```
用户提问
“未来60分钟哪些区域可能出现VIL超过133的强回波，分别持续多久？”
        ↓
Qwen识别任务类型
发现问题包含“未来、区域、持续时间”
        ↓
调用 forecast_hazards 工具
参数：
event_id = 当前事件
model_name = cascast
threshold = 133
lead_minutes = 60
        ↓
读取预先保存的 CasCast 数据
13帧历史观测
10路 ensemble × 12帧未来预测
ensemble mean
12帧 ground truth
        ↓
Python现场分析
对 ensemble mean 的12帧预测：
1. 筛选 VIL ≥ 133 的像素
2. 合并空间连通区域
3. 过滤小于16像素的区域
4. 得到 H1、H2、H3
5. 将像素位置近似映射为经纬度
6. 逐帧检查各区域是否仍超过阈值
7. 计算开始、结束和连续持续时间
8. 与 ground truth 比较，计算 CSI、POD、FAR
        ↓
工具返回结构化结果
区域、坐标、面积、起止时效、持续时间和评价指标
        ↓
Qwen生成中文回答
并根据系统提示补充适用范围和限制
        ↓
网页展示文字、空间图像和调用记录
```



## TASK3 Ensemble不确定性：分析当前事件10路CasCast ensemble的不确定性，比较位置、面积和持续时间。

### Pixel-wise uncertainty

- 像素标准差均值：整体不确定性水平

对每个像素、每个预测时效，系统都有10个成员的VIL预测：

```
M1, M2, M3, …, M10
```

先计算该位置的成员标准差：


$$
\sigma_{t,y,x} = \sqrt{\frac{1}{10}\sum_{m=1}^{10} (VIL_{m,t,y,x}-\overline{VIL}_{t,y,x})^2}
$$


再对12个时效和全部384×384像素取平均：
$$
MeanStd = mean(\sigma_{t,y,x})
$$
大部分背景像素变化较小，会把局部强回波区域的高分歧“平均掉”。所以还需要P95。

- 像素标准差P95：局部高不确定性
- 检测到对象的成员数

- 对象中心离散度

系统对每个成员取最大的强回波连通区域，并计算其质心：
$$
 c_m=(x_m,y_m)
$$
然后计算10个质心相对于平均质心的空间离散程度。可以直观理解为：
$$
Spread = \sqrt{ mean[(x_m-\bar{x})^2+(y_m-\bar{y})^2] }
$$

### Storm-object uncertainty

**中图：Pixel-wise std · 60分钟内最大**

每个像素在10个成员之间的标准差，并取未来12帧中的最大值

**右图：超阈值概率 · 60分钟内最大**

对每个像素、每个时效，统计10个成员中有多少成员预测 `VIL≥当前等级对应阈值`，再取未来60分钟内的最大值。

**Hazard Overview**

每个像素保留未来一小时中出现过的最高ensemble mean VIL

```
用户提问
        ↓
Qwen判断为ensemble不确定性任务
        ↓
调用 analyze_ensemble_uncertainty
参数：event_id、threshold、minimum_region_pixels=16
        ↓
读取提前保存的真实CasCast推理包
history:          13×384×384
ensemble_members: 10×12×384×384
ensemble_mean:    12×384×384
ground_truth:     12×384×384
        ↓
Python现场计算
像素标准差、超阈值成员比例
每个成员的强回波对象、面积、中心和活跃时长
ensemble mean的H1/H2/H3
与ground truth比较的CSI/POD/FAR/MSE/MAE
        ↓
返回结构化JSON
        ↓
Qwen根据JSON生成分析文字
        ↓
前端按需读取数组并渲染
均值图、最大标准差图、最大概率图
10张成员图、Hazard时间合成图
```



## 7. 展示10事件整体评价（2分钟）

选择“10事件整体评价”。

重点展示：

- CasCast与Persistence对比。
- 多阈值CSI。
- +5到+60分钟逐时效曲线。
- Ensemble可靠性。
- Storm-object评价。

重点结论：

> 在VIL≥133下，CasCast CSI为0.499，Persistence为0.333；CasCast RMSE为16.70，Persistence为26.55。但CasCast CSI从+5分钟的0.747下降到+60分钟的0.321，说明预测能力随时效下降。

继续补充你的思考：

> 标准差与真实误差相关系数为0.585，说明ensemble分歧有信息；但80%区间实际只覆盖49.1%，说明ensemble可能低估不确定性。

强调：

> 这是当前10个事件的结果，不是完整SEVIR测试集结论。

## 8. 展示Zephyrus工具迁移（1分钟）

依次简单展示，不要深入：

- Data Indexer：按事件、年份、月份、storm type查询CATALOG。
- Historical Statistics：统计年份、月份、storm type和`data_max`。
- Geolocator：返回事件覆盖框和近似中心。
- History Comparison：与历史同类记录比较。

随后主动说明完成边界：

> 这些目前是基础原型。Data Indexer主要读取CATALOG metadata；Geolocator还没有行政区反向编码；统计工具还不是完整HDF5逐帧统计。

## 9. 最后展示上传功能（1分钟）

先上传单帧官方VIL示例：

> 这张图有哪些VIL超过133的强回波区域？

说明单帧只能分析当前状态。

如果已经准备好13张连续图片，再展示序列上传：

- 历史强回波演变。
- Storm-object历史轨迹。
- Persistence未来基线。

明确说明：

> 13帧上传目前还不能现场运行CasCast，因为还缺EarthFormer确定性预测和latent压缩链路。

## 10. 用局限和下一步收尾（1分钟）

最后列四点即可：

1. 当前只有10个事件，不是完整测试集。
2. CasCast结果是提前运行并保存的，不是任意事件现场推理。
3. Zephyrus工具只完成基础适配。
4. Storm-object尚未完整处理对象分裂和合并。





| NOAA事件类型      | VIL记录数 |
| ----------------- | --------- |
| Thunderstorm Wind | 1,833     |
| Hail              | 1,043     |
| Flash Flood       | 385       |
| Tornado           | 194       |
| Flood             | 177       |
| Funnel Cloud      | 102       |
| Heavy Rain        | 94        |
| Lightning         | 82        |

但是，当前保存CasCast预测的10个事件都是随机 `R...` 事件

| 类型              | 与VIL强回波关系                         | 是否适合当前Demo     |
| ----------------- | --------------------------------------- | -------------------- |
| Thunderstorm Wind | 强对流可能伴随雷暴大风                  | 适合                 |
| Hail              | 强回波经常与冰雹相关                    | 适合                 |
| Tornado           | 可能伴随强对流，但仅靠VIL不能识别龙卷风 | 可作为辅助标签       |
| Funnel Cloud      | 与旋转对流有关，但样本较少              | 可作为辅助标签       |
| Heavy Rain        | 强VIL可能对应强降雨                     | 适合，但需降雨量验证 |
| Flash Flood       | 与累计降雨、地形和水文条件有关          | 当前不宜直接判断     |
| Flood             | 时间尺度更长，不能仅依靠60分钟VIL       | 当前不适合           |
| Lightning         | 与对流有关，但VIL不能直接预测闪电       | 不宜直接判断         |



你的系统目前已经有“LLM调用气象工具”的基础，但与Zephyrus的核心差距是：**Qwen主要在做工具路由和结果解释，还没有自主编写分析代码、观察中间结果并修改方案。**

可以按下面优先级改进，不需要一次全部完成。

## 整理SEVIRWorld工具环境

| 组件                  | 能力                                                         |
| --------------------- | ------------------------------------------------------------ |
| `SevirDataIndexer`    | CATALOG 查询、事件 metadata、HDF5 读取、CasCast 缓存读取     |
| `CasCastForecaster`   | 读取 13 帧历史和已保存的 10 路 ensemble 预测，执行强回波分析 |
| `SevirGeolocator`     | 像素转经纬度、事件覆盖范围、离线州县定位                     |
| `SevirStatistics`     | 年份、月份、storm type 统计及历史强度比较                    |
| `UncertaintyAnalyzer` | pixel-wise 不确定性、对象轨迹、跨 ensemble 对象匹配          |
| `VerificationMetrics` | 多阈值 CSI、POD、FAR、HSS、FSS                               |
| `Visualizer`          | 逐时效指标图、对象轨迹图、storm type 图和 ensemble 共识图    |

| Zephyrus实现   | SEVIRWorld实现        | 主要变化                                                     |
| -------------- | --------------------- | ------------------------------------------------------------ |
| `DataIndexer`  | `SevirDataIndexer`    | WeatherBench2 Zarr 按时间/变量读取，改为 SEVIR CATALOG、HDF5、NPZ 缓存和上传数据读取 |
| `Geolocator`   | `SevirGeolocator`     | 全球经纬度网格与 Natural Earth，改为 SEVIR 投影像素转换及 Census 州县定位 |
| `Forecaster`   | `CasCastForecaster`   | Stormer 全球天气预报，改为 CasCast 雷达临近预测及强回波分析  |
| `Climatology`  | `SevirStatistics`     | WB2 气候态统计，改为 SEVIR 年/月/storm type 和历史事件比较   |
| 无独立对应工具 | `UncertaintyAnalyzer` | 新增 CasCast ensemble 不确定性、对象追踪和跨成员匹配         |
| 无独立对应工具 | `VerificationMetrics` | 新增 CSI、HSS、FSS 等雷达临近预测指标                        |
| 无独立对应工具 | `Visualizer`          | 将轨迹图、指标图、ensemble 共识图封装为 Agent 工具           |
| `Simulator`    | 暂未迁移              | Zephyrus 使用 JAX-GCM 全球气候模拟，不适合直接迁移到 SEVIR 雷达临近预测 |

- [ ] `SevirDataIndexer`
  - 精确事件查询
  - CATALOG检索
  - HDF5事件读取
  - CasCast缓存读取

- [ ] `SevirGeolocator`
  - 像素到经纬度
  - 覆盖范围
  - 后续增加州、县反向地理编码

- [ ] `CasCastForecaster`
  - 输入13帧历史VIL
  - 现场运行CasCast
  - 返回各ensemble成员和ensemble mean
  - 当前预计算NPZ可以作为缓存模式保留

- [ ] `SevirStatistics`
  - metadata统计
  - HDF5逐帧统计
  - 按storm type、月份、年份比较

- [ ] `UncertaintyAnalyzer`
  - pixel-wise uncertainty
  - storm-object uncertainty
  - ensemble可靠性分析

- [ ] `Visualizer`
  - 帧图、概率图、轨迹图和柱状图

每个API需要写清楚docstring，包括输入、输出、单位、数据限制。之后让Qwen在上下文中看到这些说明，这才接近ZephyrusWorld。

## 第二阶段：加入代码执行能力

这是最关键的差距。

当前：

```text
Qwen选择一个固定函数
→ 函数返回固定JSON
```

目标：

```text
Qwen根据问题生成Python分析代码
→ 代码调用SevirWorld API
→ 执行并返回结果
```

例如用户问：

> 这个事件未来60分钟最大的强回波区域在哪里，与同类型历史事件相比是否异常？

Agent可以生成：

```python
forecast = forecaster.run(event_id)
regions = uncertainty.detect_objects(
    forecast["ensemble_mean"],
    threshold=133
)
location = geolocator.pixel_to_latlon(
    event_id,
    regions[0]["centroid"]
)
history = statistics.compare_by_storm_type(
    event_id,
    metric="max_vil"
)
```

这里不是为每个问题提前写好一个`forecast_hazards`函数，而是Agent自己组合多个基础API。

## 第三阶段：实现安全执行服务器

不能直接在后端用`exec()`运行Qwen生成的任意代码，需要增加受控环境：

- [ ] 独立代码执行进程
- [ ] 只允许导入指定库
- [ ] 只注入SEVIRWorld工具
- [ ] 禁止任意文件访问
- [ ] 禁止网络和系统命令
- [ ] 设置运行超时
- [ ] 设置内存与输出长度限制
- [ ] 捕获标准输出、异常和图片
- [ ] 保存每次执行代码和结果

这一层对应Zephyrus的Code Execution Server。

## 第四阶段：先实现Zephyrus-Direct

Direct模式相对简单：

```text
问题
→ Qwen一次生成完整Python代码
→ 执行
→ 如果报错，把错误返回Qwen
→ Qwen修改代码
→ 成功后根据结果回答
```

建议最多允许3至5次修正，不需要立即照搬论文的20次。

验收标准：

- [ ] 能完成至少5种自然语言问题
- [ ] 能在一次代码中组合两个以上工具
- [ ] 参数或代码出错后能够自动修正
- [ ] 最终答案必须来自执行结果
- [ ] 页面展示生成代码、工具调用和结果

## 第五阶段：再实现Reflective模式

Reflective模式允许多轮分析：

```text
生成一小段代码
→ 执行
→ 查看Observation
→ 判断结果是否合理
→ 决定下一步
→ 再执行
→ 最终回答
```

例如：

```text
第一步：查看ensemble整体差异
第二步：发现+30分钟不确定性最大
第三步：只分析该时刻的storm objects
第四步：定位争议最大的对象
第五步：与Ground Truth或历史事件比较
```

需要保存：

- `thought/plan`：下一步分析计划
- `code`：本轮代码
- `observation`：执行结果
- `error`：异常信息
- `final_answer`：最终结论

不必向网页暴露模型的内部思维，只展示简短的“分析步骤”和实际执行记录。

## 第六阶段：减少固定任务限制

当前Agent仍受预设任务约束。可以逐步支持组合问题：

- [ ] 查询＋预测
- [ ] 预测＋地理定位
- [ ] 预测＋不确定性
- [ ] 预测＋Ground Truth验证
- [ ] 当前事件＋历史同类事件比较
- [ ] 不同阈值敏感性分析
- [ ] 多个事件横向比较
- [ ] 用户上传13帧后现场分析和预测

前端可以提供问题示例，但不应把用户限制在几个固定按钮中。

## 第七阶段：补强气象工具

为了接近ZephyrusWorld，还需要补齐：

- [ ] 任意官方HDF5事件读取验证
- [ ] 行政区反向地理编码
- [ ] 逐帧storm-object追踪
- [ ] 跨ensemble对象匹配
- [ ] storm type条件统计
- [ ] 多阈值CSI/HSS/FSS
- [ ] 用户上传13帧后的现场CasCast推理
- [ ] Ground Truth存在时自动评价
- [ ] 图表作为Agent可调用工具

JCM模拟器不必强行迁移。SEVIR与全球天气场不同，目前可以把“模拟器”标记为可选扩展，而不是核心缺口。

## 第八阶段：建立Agent评测

不能只展示网页，还要评价Agent本身：

- [ ] 工具选择正确率
- [ ] 代码执行成功率
- [ ] 错误自动修复率
- [ ] 多工具组合成功率
- [ ] 最终数值答案正确率
- [ ] 是否引用了真实工具结果
- [ ] 平均调用轮数
- [ ] 响应时间和API成本

再比较：

```text
固定工具路由
vs
Direct代码Agent
vs
Reflective代码Agent
```

这会让项目从“网页Demo”提升到具有实验内容的Agent研究。

## 推荐实施顺序

结合你目前的时间，近期只做下面五项最有价值：

1. 将现有函数整理为`SEVIRWorld`基础API。
2. 增加安全Python执行环境。
3. 实现Zephyrus-Direct的生成、执行、报错修正闭环。
4. 准备5至10个必须组合多个工具的问题。
5. 比较固定工具模式与Direct模式的成功率。

完成后，你可以准确地向老师汇报：

> 当前版本不仅能由Qwen选择预设工具，还增加了面向SEVIR的代码生成与执行闭环。Agent可以根据自然语言问题自主组合数据索引、CasCast预测、地理定位和不确定性分析API，并根据执行错误进行修正。Reflective多轮分析和任意上传序列现场推理是下一阶段工作。

最重要的是：**先完成Direct，不要同时改现场CasCast、Reflective、完整HDF5和所有气象分析功能。**否则每一部分都容易停在半成品。
按你当前已有的10个NPZ事件、CasCast权重和CATALOG，除了“任意上传序列现场运行CasCast”以外，其余大部分都能继续实现，只是完成程度不同。

| 功能                 | 当前状态                   | 现在能否实现     | 主要条件                 |
| -------------------- | -------------------------- | ---------------- | ------------------------ |
| 官方HDF5读取验证     | 代码已写，合成文件测试通过 | 可以             | 下载至少一个官方HDF5     |
| 行政区反向地理编码   | 只有经纬度                 | 可以             | 增加行政区边界数据       |
| 逐帧storm-object追踪 | 已有基础版                 | 可以改进         | 当前NPZ已经够用          |
| 跨ensemble对象匹配   | 尚未真正完成               | 可以             | 当前10路ensemble已经够用 |
| storm type条件统计   | metadata基础版已完成       | 可以扩展         | 像素统计需要HDF5         |
| 多阈值CSI/HSS/FSS    | 多阈值CSI部分已有          | 可以             | 当前预测和GT已经够用     |
| 上传13帧现场CasCast  | 目前只有Persistence        | 暂时不能直接完成 | 缺EarthFormer现场推理链  |
| GT存在时自动评价     | 10个事件已经完成           | 基本完成         | 上传场景需同时提供GT     |
| 图表作为Agent工具    | 目前只是前端固定展示       | 可以             | 封装绘图函数即可         |

## 1. 官方HDF5事件读取验证

**现在可以做，但需要下载一个HDF5文件。**

当前 [load_sevir_vil_event](D:/Profile/专业实习/sevir-agent-demo/sevir_tools.py:502)已经可以：

```text
event_id
→ CATALOG查file_name和file_index
→ HDF5读取vil[file_index]
→ 识别49帧时间轴
→ 返回逐帧统计
```

现在缺的是一次真正的官方文件验证。只下载一个与测试事件对应的HDF5即可，不需要完整SEVIR。

预计：下载完成后约半天。

## 2. 行政区反向地理编码

**现在可以做。**

目前系统只能得到：

```text
43.28°N, 90.75°W
```

可以增加行政区边界数据，然后完成：

```text
经纬度
→ 国家
→ 州
→ 县
```

推荐优先使用离线边界数据，不依赖在线API。城市级定位不能只靠行政区多边形，需要额外城市数据，而且雷达像素中心也未必位于城市范围。

预计：国家/州基础版半天，州县完整版1至2天。

## 3. 逐帧storm-object追踪

**已有基础版，可以立即改进。**

现有 [_storm_tracks](D:/Profile/专业实习/sevir-agent-demo/sevir_tools.py:1813)已经做了：

```text
逐帧连通域
→ 根据边界框重叠或中心距离匹配
→ 形成轨迹
```

但现在还是简单贪心匹配，需要改进：

- 使用真实连通域IoU，而不是外接矩形IoU；
- 使用匈牙利算法进行一对一匹配；
- 距离从像素换成km；
- 允许短暂消失一帧；
- 处理对象分裂与合并；
- 返回每帧面积、强度、中心和速度。

不需要新数据，当前10个事件就能开发和测试。

预计：基础改进1天，分裂合并再增加1至2天。

## 4. 跨ensemble对象匹配

**目前没有真正完成，但现在可以实现。**

现有不确定性分析主要比较每个成员最大的时空合并区域，不能保证比较的是同一场风暴。

需要改成：

```text
每个成员先生成逐帧轨迹
→ 比较不同成员轨迹的时间重合、空间重合和路径距离
→ 聚类为同一个共识风暴对象
```

之后输出：

- 10个成员中有几个预测到它；
- 出现概率/成员支持率；
- 中心位置离散度；
- 面积Q10、Q50、Q90；
- 持续时间分布；
- 移动路径差异。

当前10路ensemble完全够用，不需要额外下载。

预计：2至4天，是这批任务中技术难度最高但最符合老师要求的一项。

## 5. Storm type条件统计

**metadata版本已经有了，可以马上扩展。**

当前 [historical_sevir_statistics](D:/Profile/专业实习/sevir-agent-demo/sevir_tools.py:604)已经可以按`event_type`筛选并统计：

- 事件数量；
- 年份和月份分布；
- `data_max`均值与分位数；
- 最大值排名。

现在可以增加：

- 不同storm type横向对比；
- 每月各类型占比；
- Hail、Tornado、Flash Flood的`data_max`箱线图；
- 当前事件在同类型事件中的经验分位数。

这些只需要CATALOG。

但如果要比较“不同storm type的逐帧强回波面积和持续时间”，就必须读取对应HDF5，不能只靠metadata。

预计：metadata扩展半天，像素级统计取决于HDF5数量。

## 6. 多阈值CSI/HSS/FSS

**现在可以实现，数据已经够用。**

当前已经保存CasCast预测和Ground Truth，可以直接计算：

```text
VIL阈值：16、74、133、160、181、219
```

目前项目已有多阈值CSI评价基础，但还缺：

- HSS；
- FSS；
- 每个预测时效的指标；
- 10事件汇总均值；
- CasCast与Persistence对比。

FSS还要指定邻域尺度，例如：

```text
1、4、8、16、32像素
```

这项非常适合近期完善，因为不需要模型重跑。

预计：1至2天，包括测试和图表。

## 7. 用户上传13帧后现场CasCast推理

**目前不能直接完成，是主要阻塞项。**

当前上传13帧后执行的是：

```text
最后一帧重复12次
```

即Persistence，不是CasCast。

真实CasCast链路需要：

```text
上传13帧
→ 按训练方式预处理
→ EarthFormer生成确定性未来
→ Autoencoder压缩
→ CasCast扩散模型生成ensemble latent
→ Autoencoder解码
→ 返回未来12帧
```

你目前有：

- CasCast概率模型权重；
- Autoencoder权重。

但没有可直接调用的EarthFormer确定性模型checkpoint。因此不能只把上传图片送入现有CasCast权重。

这项要最后做，预计至少3至7个工作日，而且前提是补齐EarthFormer checkpoint和现场预处理流程。

## 8. Ground Truth存在时自动评价

**对10个缓存事件已经基本完成。**

当前NPZ中存在Ground Truth时，系统已经自动计算：

- CSI；
- POD；
- FAR；
- MSE/RMSE；
- MAE；
- 基础storm-object验证。

还需要做的主要是：

- 加入HSS和FSS；
- 统一单事件与数据集评价接口；
- 上传序列时允许用户额外上传未来12帧GT；
- 明确没有GT时不显示准确率。

所以这项不需要从头重做。

## 9. 图表作为Agent可调用工具

**现在可以实现。**

目前图表逻辑主要固定在前端。真正的Agent工具应该类似：

```python
plot_lead_time_metrics(...)
plot_ensemble_member_comparison(...)
plot_storm_tracks(...)
plot_object_uncertainty(...)
```

返回：

```json
{
  "chart_type": "storm_tracks",
  "image_url": "/artifacts/xxx.png",
  "data": {}
}
```

这样以后Qwen或代码Agent可以根据问题决定画什么图，而不是每次固定展示全部图表。

预计：先封装3至4种图表约1天。

## 当前最合理的顺序

按你每天约4小时安排：

1. 多阈值CSI、HSS、FSS。
2. 改进逐帧storm-object追踪。
3. 实现跨ensemble对象匹配。
4. 将不确定性图表封装为Agent工具。
5. 扩展storm type条件统计。
6. 下载一个HDF5完成官方读取验证。
7. 增加基础行政区反向定位。
8. 最后处理上传13帧现场CasCast推理。

前六项都能利用现有数据推进。**不建议现在先攻现场CasCast推理**，因为它依赖尚未补齐的EarthFormer链路，会占用大量时间，却不会立即改善你现有10个事件的分析质量。
【【【读一下上面的内容，然后完成下面的任务】】】第一阶段：整理成SEVIRWorld工具环境 第二阶段：加入代码执行能力 第三阶段：实现安全执行服务器 第四阶段：先实现Zephyrus-Direct 第六阶段：减少固定任务限制 第七阶段：补强气象工具（行政区反向地理编码、逐帧storm-object追踪、跨ensemble对象匹配、Storm type条件统计、多阈值CSI/HSS/FSS、图表作为Agent可调用工具
