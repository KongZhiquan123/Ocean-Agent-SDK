# 后台训练模式

## 工作原理

训练任务在后台执行，`ocean_sr_train_start` 启动后会等待 `training_start` 事件（最长 5 分钟），
如果启动阶段崩溃会立即返回错误详情，否则返回启动成功信息：

```
ocean_sr_train_start(...)
  → 内部等待 training_start 事件（数据加载 + 模型构建）
  → 返回 { status: "started", process_id: "train-xxx-xxx", ... }
  → 若启动阶段崩溃: 返回 { status: "error", error_summary: {...}, ... }
```

---

## 状态查询工具 ocean_sr_train_status

| 操作 | 调用方式 | 说明 |
|------|----------|------|
| 查询状态 | `ocean_sr_train_status({ process_id: "xxx" })` | 返回运行状态、耗时、进度等 |
| 查看日志 | `ocean_sr_train_status({ action: "logs", process_id: "xxx", tail: 50 })` | 返回最后 50 行日志 |
| 增量日志 | `ocean_sr_train_status({ action: "logs", process_id: "xxx", offset: 12345 })` | 从上次位置继续读取 |
| 终止训练 | `ocean_sr_train_status({ action: "kill", process_id: "xxx" })` | 发送终止信号 |
| 列出所有 | `ocean_sr_train_status({ action: "list" })` | 列出所有训练进程 |
| 等待变化 | `ocean_sr_train_status({ action: "wait", process_id: "xxx", timeout: 120 })` | 长轮询等待训练状态变化（超时秒数） |

---

## 训练状态

| 状态 | 含义 |
|------|------|
| `running` | 训练进行中（附带 progress: 当前 epoch / 总 epoch / 预估剩余时间） |
| `completed` | 训练成功完成（exit code = 0） |
| `failed` | 训练失败（附带 error_summary: 错误类型 / 消息 / 建议） |
| `killed` | 被用户或系统终止 |

---

## 重要行为准则

1. **启动后立即 wait**：训练启动后，立即调用 `ocean_sr_train_status({ action: "wait", process_id, timeout: 120 })`，等 2 分钟捕获快速完成或早期崩溃
2. **用户询问时再次 wait**：当用户询问训练进度时，用 wait 模式等 2 分钟，如果训练恰好在这 2 分钟内完成/失败，用户立刻看到结果
3. **失败时展示分析**：失败时从 `error_summary` 提取错误类型和建议，展示给用户并询问是否调整参数重试
4. **保持对话可用**：训练在后台运行，Agent 可以继续回答用户的其他问题
5. **主动状态检查**：如果之前启动过训练进程，收到用户新消息时先调用 `action="list"` 检查训练状态
6. **服务器关闭自动清理**：如果服务器关闭，训练进程会被自动终止

---

## 训练启动后的标准流程

```
1. ocean_sr_train_start 返回 status="started"
   → 告知用户训练已启动

2. 立即调用 ocean_sr_train_status({ action: "wait", process_id, timeout: 120 })
   → 等待 2 分钟

3. 根据返回结果：
   - process_status="completed" → 主动询问是否生成可视化和报告
   - process_status="failed" → 展示 error_summary + suggestions，询问是否重试
   - process_status="running"（超时）→ 告知用户训练仍在运行（含进度），等待用户后续询问
```

---

## 训练完成后的处理

当 `ocean_sr_train_status` 返回 `process_status="completed"` 时：

1. 展示最终测试指标
2. **主动询问**是否生成可视化和报告
3. 等待用户确认后执行

## 训练失败后的处理

当 `ocean_sr_train_status` 返回 `process_status="failed"` 时：

1. 从 `error_summary` 中提取错误类型 (`failureType`) 和建议 (`suggestions`)
2. 展示给用户："训练失败原因是 XXX，建议 YYY"
3. 询问用户是否调整参数重试，**不自动重试**

详见 `references/visualization.md`
