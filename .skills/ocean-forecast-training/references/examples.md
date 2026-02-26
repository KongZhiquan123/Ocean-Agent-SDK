# 对话示例

## 示例 1: 完整训练流程

```
用户: 我想用预处理好的海洋数据训练一个预报模型
Agent: 好的！我先帮您检查 GPU 环境。
      [调用 ocean_forecast_check_gpu]
      检测到 1 张 NVIDIA A100 (40GB)，可用 38.5GB。

用户: 数据在 /data/ocean/preprocessed，输出放到 /data/ocean/forecast_output
Agent: 收到，我来验证数据集。
      [调用 ocean_forecast_train({ dataset_root: "/data/ocean/preprocessed", log_dir: "/data/ocean/forecast_output" })]

      数据集验证结果：
      - 变量: uo, vo, thetao
      - 空间尺寸: 128 × 128
      - 训练集: 3650 个时间步
      - 验证集: 365 个时间步
      - 测试集: 365 个时间步
      - 时间范围: 1993-01-01 ~ 2003-12-31

      请选择训练模型。推荐 FNO2d（频谱方法，适合捕获全局模式）。
      [调用 ocean_forecast_list_models]

用户: 用 FNO2d
Agent: [调用 ocean_forecast_train({ ..., model_name: "FNO2d" })]

      请确认训练参数：
      - 时序参数: in_t=7, out_t=1, stride=1（7 天历史预测 1 天）
      - 训练参数: epochs=300, lr=0.001, batch_size=4
      - GPU: A100 (0号), use_amp=false（FNO2d 关闭 AMP）
      - gradient_checkpointing=true

      需要调整参数吗？

用户: in_t 改成 14 天，其他参数确认
Agent: [调用 ocean_forecast_train({ ..., in_t: 14, user_confirmed: true, confirmation_token: "xxx" })]

      训练已启动！进程 ID: forecast-abc123
      - 模型: FNO2d
      - in_channels: 42 (14 天 × 3 变量)
      - out_channels: 3 (1 天 × 3 变量)

      让我等待训练状态...
      [调用 ocean_forecast_train_status({ action: "wait", process_id: "forecast-abc123", timeout: 120 })]
```

## 示例 2: predict 模式

```
用户: 训练完成了，帮我对测试集做预测
Agent: [调用 ocean_forecast_train({ mode: "predict", dataset_root: "/data/ocean/preprocessed", log_dir: "/data/ocean/forecast_output", model_name: "FNO2d" })]

      推理已启动！进程 ID: predict-xyz789
      正在加载 best_model.pth 对测试集执行预测...

      [调用 ocean_forecast_train_status({ action: "wait", process_id: "predict-xyz789", timeout: 300 })]

      推理完成！共生成 365 个预测样本，保存在:
      /data/ocean/forecast_output/predictions/

      是否生成预测可视化？

用户: 好的
Agent: [调用 ocean_forecast_visualize({ log_dir: "/data/ocean/forecast_output", mode: "predict", dataset_root: "/data/ocean/preprocessed" })]

      已生成 10 个可视化图表：
      - predict_sample_0_var_uo.png (经向流速)
      - predict_sample_0_var_vo.png (纬向流速)
      - predict_overview.png (总览)
      ...
```

## 示例 3: 训练失败重试

```
Agent: 训练启动后 30 秒报错：
      错误类型: CUDA OOM
      错误信息: CUDA out of memory. Tried to allocate 2.5 GiB
      建议:
        1. 减小 batch_size: 4 → 2
        2. 减小 in_t: 14 → 7

      是否调整参数重试？

用户: batch_size 改成 2 重试
Agent: [重新调用 ocean_forecast_train，batch_size=2]
      训练已重新启动...
```
