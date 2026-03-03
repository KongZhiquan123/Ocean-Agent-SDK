# /ocean-api-change — API/SSE 变更检查清单

当修改 HTTP API 接口、SSE 事件格式、或会话管理逻辑时，按以下清单逐项检查。

## 触发条件

以下任一文件被修改时，必须执行此清单：
- `src/server.ts`
- `src/agent-manager.ts`（特别是 `convertProgressToSSE` 和 `buildDirectives`）
- `src/conversation-manager.ts`

## 检查清单

### 1. API 请求契约
- [ ] `src/server.ts` 请求体校验（message, mode, outputsPath, context）是否正确
- [ ] 新增的必填字段是否在 `test-client.ts` 的 `chat()` 函数中同步添加
- [ ] 新增的可选字段是否有合理的默认值
- [ ] `X-API-Key` 认证中间件是否影响

### 2. SSE 事件格式
- [ ] `src/agent-manager.ts` 中 `convertProgressToSSE()` 的事件类型是否完整
- [ ] SSE 事件格式：`{type, timestamp, ...payload}` 结构是否一致
- [ ] `test-client.ts` 的 SSE 解析 switch-case（~line 147-206）是否覆盖所有事件类型
- [ ] `heartbeat` 间隔（2s）和 `timeout`（10min）是否合理

### 3. 会话管理
- [ ] `src/conversation-manager.ts` 中 agentId 校验格式：`/^agt-[a-zA-Z0-9_-]+$/`
- [ ] 会话超时时间（30min）、清理间隔（5min）、最大会话数（100）是否需要调整
- [ ] `Agent.resumeFromStore()` 是否能正确恢复变更后的 agent 状态
- [ ] `.kode/{agentId}/` 目录下的持久化数据格式是否兼容

### 4. 安全与权限
- [ ] `DANGEROUS_PATTERNS` 黑名单是否需要更新
- [ ] `SAFE_READ_PATTERNS` / `SAFE_WRITE_PATTERNS` 是否需要扩展
- [ ] sandbox `allowPaths` 配置是否正确（`/data`, `.skills`, `outputsPath`）
- [ ] `buildDirectives()` 中的 outputsPath 约束是否生效

### 5. 文档同步
- [ ] `CLAUDE.md` 中 API Usage 部分是否更新
- [ ] `CLAUDE.md` 中 SSE Event Types 列表是否更新
- [ ] README 中 API 调用示例是否同步

### 6. 最终验证
- [ ] `npm run typecheck` 通过
- [ ] `npm run test:client` 通过
- [ ] 启动服务后手动发一次请求，确认 SSE 流正常

## 常见遗漏

- 修改了 SSE 事件格式但 test-client.ts 的解析逻辑未更新，导致事件静默丢失
- 新增 API 字段后忘记更新 CLAUDE.md 的文档
- 修改了 buildDirectives() 但 outputsPath 约束意外放宽
- conversation-manager 的 session 恢复逻辑与新字段不兼容
