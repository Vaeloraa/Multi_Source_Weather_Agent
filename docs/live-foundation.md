# 现场 ERA5/Aurora 模式

现场模式不依赖 `data/era5_cases/<event_id>`。请求由 UTC 时间、经纬度（或区域）和变量配置确定，并以 `request_key` 缓存到临时 artifact 目录。

## 前端

在“现场环境参数”中填写：

- UTC 时间，例如 `2019-08-01T21:00:00Z`；
- 纬度和经度；
- 询问 ERA5/Aurora 环境问题。

可以同时上传一张或 13 张 SEVIR VIL 图。图像只作为局地证据，不会替代现场 ERA5 时间和地点。

## API

`POST /api/run` 可发送：

```json
{
  "question": "该地点未来大尺度环境趋势如何？",
  "data_policy": "live",
  "data_source": "live",
  "analysis_time_utc": "2019-08-01T21:00:00Z",
  "latitude": 34.75,
  "longitude": -84.75,
  "forecast_leads": [6, 12, 18, 24],
  "trace": true
}
```

执行顺序是：请求绑定 → Unified Agent 按需调用 `foundation.fetch_era5` 或 `foundation.forecast_aurora` → 根据 Observation 继续分析或结束。缺时间/地点、CDS 取数失败、Aurora 依赖缺失和模型推理失败都会保留在 `trace` 的对应阶段，绝不把缺失值改成 `0.0`。

## 外部依赖

现场 ERA5 取数需要安装 `cdsapi` 并在用户目录配置 CDS 凭据；Aurora 需要 `aurora`、PyTorch 和可用的 checkpoint。没有这些依赖时，系统仍会返回明确的可恢复错误和 request key，不会回退到旧 event case。
