# 后台训练模式

## 工作原理

训练任务在后台执行，`ocean_sr_train` 启动后立即返回，不会阻塞对话：

```
ocean_sr_train(...)
  → 返回 { status: "started", process_id: "train-xxx-xxx", ... }
  → 训练在后台持续运行
  → Agent 可以继续与用户对话
```

---

## 状态查询工具 ocean_sr_train_status

| 操作 | 调用方式 | 说明 |
|------|----------|------|
| 查询状态 | `ocean_sr_train_status({ process_id: "xxx" })` | 返回运行状态、耗时等 |
| 查看日志 | `ocean_sr_train_status({ action: "logs", process_id: "xxx", tail: 50 })` | 返回最后 50 行日志 |
| 增量日志 | `ocean_sr_train_status({ action: "logs", process_id: "xxx", offset: 12345 })` | 从上次位置继续读取 |
| 终止训练 | `ocean_sr_train_status({ action: "kill", process_id: "xxx" })` | 发送终止信号 |
| 列出所有 | `ocean_sr_train_status({ action: "list" })` | 列出所有训练进程 |

---

## 训练状态

| 状态 | 含义 |
|------|------|
| `running` | 训练进行中 |
| `completed` | 训练成功完成（exit code = 0） |
| `failed` | 训练失败（exit code != 0） |
| `killed` | 被用户或系统终止 |

---

## 重要行为准则

1. **启动后不要主动轮询**：训练启动后，告知用户训练已开始，然后等待用户的下一步指令
2. **用户询问时才查询**：只有当用户主动询问训练进度时，才调用 `ocean_sr_train_status`
3. **保持对话可用**：训练在后台运行，Agent 可以继续回答用户的其他问题
4. **服务器关闭自动清理**：如果服务器关闭，训练进程会被自动终止

---

## 训练启动后的标准回复

```
训练已启动！

- 进程 ID: train-1707123456-abc123
- 日志目录: /path/to/log_dir

您可以：
- 查看训练状态：告诉我"查看训练状态"
- 查看最新日志：告诉我"查看训练日志"
- 终止训练：告诉我"终止训练"

训练在后台运行，您可以继续与我对话。
```

---

## 训练完成后的处理

当 `ocean_sr_train_status` 返回 `process_status="completed"` 时：

1. 展示最终测试指标
2. **主动询问**是否生成可视化和报告
3. 等待用户确认后执行

详见 `references/visualization.md`
