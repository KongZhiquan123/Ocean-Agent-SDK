# KODE-HUD 页面信息架构

## 1. 总体结构

建议页面分为四大区块：

1. 顶部全局状态栏
2. 中部 Agent 看板
3. 右侧 Agent 详情面板
4. 底部实时事件流

目标不是像聊天应用，而是像一套轻量版 Mission Control。

## 2. 顶部全局状态栏

顶部应该始终显示任务级摘要。

建议字段：

- 当前任务名
- 活跃 Agent 数
- Running 数
- Waiting 数
- Error 数
- Done 数
- 最近事件时间
- 总运行时长
- 告警数

建议视觉：

- 左侧任务标题
- 中间状态计数
- 右侧连接状态、更新时间、过滤器

## 3. 中部 Agent 看板

这是页面主体。

每个 Agent 一张卡片，像看板一样平铺显示。

### 每张卡片最少字段

- `role`
- `agentId`
- `status`
- `currentAction`
- `currentTarget`
- `lastOutput`
- `lastHeartbeatAt`
- `runtime`

### 推荐展示文案

例如：

- `Researcher`
- `Running`
- `正在读取文件`
- `src/server.ts`
- `最近产出：提取 SSE 事件结构`

### 卡片状态建议

- `idle`
- `thinking`
- `running`
- `calling_tool`
- `reading_file`
- `writing_file`
- `running_command`
- `waiting`
- `done`
- `failed`

### 卡片颜色建议

- `thinking`：浅蓝
- `running`：青绿
- `calling_tool`：橙色
- `waiting`：灰色
- `failed`：红色
- `done`：绿色

## 4. 右侧 Agent 详情面板

点击某张卡片后，右侧展开详情。

建议有 5 个标签页。

### 4.1 Live

显示：

- 当前状态
- 当前动作
- 当前对象
- 实时文本流
- 最近一次状态变化时间

### 4.2 Tools

显示：

- 工具名称
- 开始时间
- 结束时间
- 输入参数摘要
- 返回结果摘要
- 是否失败

### 4.3 Files

显示：

- 最近读取文件
- 最近写入文件
- 最近修改文件
- 最近产出文件

建议区分动作：

- `read`
- `write`
- `edit`
- `create`

### 4.4 Commands

显示：

- 执行命令
- 当前阶段：`start / stdout / stderr / end`
- 最新日志片段
- 返回码
- 运行时长

### 4.5 Artifacts

显示：

- 新生成的文件
- 报告
- 图像
- 数据文件
- notebook
- 模型权重

## 5. 底部实时事件流

这是全局时间线，不针对单卡片，而是针对整个任务。

每条事件最少包含：

- 时间
- Agent / 角色
- 事件类型
- 动作摘要
- 对象摘要

例如：

- `12:00:03 Researcher 读取文件 src/server.ts`
- `12:00:08 Coder 写入文件 src/hud-store.ts`
- `12:00:11 Reviewer 调用工具 fs_read`
- `12:00:13 Planner 更新 todo`

## 6. 关键交互

### 6.1 点击卡片

- 右侧打开详情
- 时间线自动过滤到当前 Agent

### 6.2 悬浮卡片

- 显示最近动作摘要
- 显示最近 3 个事件

### 6.3 筛选器

支持按以下维度筛选：

- 状态
- 角色
- 事件类型
- 是否报错
- 是否有新产出

### 6.4 搜索

支持搜索：

- 文件路径
- 工具名
- 命令片段
- Agent 名称
- agentId

## 7. “一直在滚动工作”的视觉要求

这个页面必须有持续更新感，但不能做成噪音墙。

建议：

- 事件流自动滚动，但允许暂停
- 每个卡片只突出当前状态和最近 1 条变化
- 新事件短暂高亮
- 长文本不要在卡片里全展开，只在详情页展开

## 8. 最重要的字段定义

### currentAction

表示当前正在做的动作。

例如：

- `正在读取文件`
- `正在写入文件`
- `正在调用工具`
- `正在运行训练`
- `正在生成报告`
- `正在等待工具结果`

### currentTarget

表示动作的对象。

例如：

- `src/agent-manager.ts`
- `main_ddp.py`
- `train.log`
- `/data/WaveCastNet/processdata`
- `fs_write`

这两个字段组成 HUD 的最小语义单元：

`谁 + 在做什么 + 对什么做`

