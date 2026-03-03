# /ocean-tool-change — Ocean 工具变更检查清单

当修改任何 `ocean_*` 工具时，按以下清单逐项检查。不可跳过。

## 触发条件

以下任一文件被修改时，必须执行此清单：
- `src/tools/ocean-SR-data-preprocess/` 下任何文件
- `src/tools/ocean-sr-training/` 下任何文件
- `src/tools/ocean-forecast-data-preprocess/` 下任何文件
- `src/tools/ocean-forecast-training/` 下任何文件

## 检查清单

### 1. 工具定义层
- [ ] `defineTool()` 中的 `params` schema 是否正确（类型、必填/可选、description）
- [ ] `exec()` 返回的 status 值是否与 SKILL.md 中记录的一致
- [ ] 如果修改了 `confirmation_token` 流程，所有阶段的 token 传递是否完整

### 2. 工具注册层
- [ ] `src/tools/index.ts` — 新工具是否已 export
- [ ] `src/config.ts` — 新工具是否在正确的 template 中注册（edit/ask 模式）
- [ ] 如果是 ask 模式工具，确认已加入 line ~225 的 ask 工具白名单

### 3. SKILL.md 文档层
- [ ] `.skills/{对应skill}/SKILL.md` 中的"可用工具"表格是否更新
- [ ] 工作流程步骤描述是否与代码一致
- [ ] "快速参数参考"表格中的参数名、类型、默认值是否同步
- [ ] `references/parameters.md` 是否更新（如果存在）

### 4. 测试层
- [ ] `test-client.ts` 中对应的测试函数是否覆盖了变更
- [ ] 运行 `npm run test:client` 确认无回归

### 5. Agent 层
- [ ] `src/agent-manager.ts` 中 `convertProgressToSSE()` 是否需要处理新的 event 类型
- [ ] `buildDirectives()` 是否需要更新 outputsPath 约束
- [ ] `DANGEROUS_PATTERNS` 是否需要更新（如果工具涉及新的 bash 命令）

### 6. 最终验证
- [ ] `npm run typecheck` 通过
- [ ] `npm run test:client` 通过
- [ ] 手动调用一次变更后的工具，确认返回格式正确

## 常见遗漏

- 修改了工具参数名但忘记更新 SKILL.md 中的参数表
- 添加了新的 status 返回值但 SKILL.md 工作流中未提及
- 修改了 confirmation_token 逻辑但测试中仍用旧 token 流程
- ask 模式下不应该暴露写操作工具
