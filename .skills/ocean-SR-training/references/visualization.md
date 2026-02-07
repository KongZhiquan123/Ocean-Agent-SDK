# 可视化与报告生成

## 训练完成后的标准流程

当 `ocean_sr_train_status` 工具返回 `process_status="completed"`时，**必须主动询问用户**：

```
训练已完成！最终测试指标：
- test_loss: 0.00123456
- MSE: 0.00012345
- PSNR: 45.67 dB
- SSIM: 0.9876

是否需要生成可视化图表和训练报告？
```

---

## 可视化工具 ocean_sr_visualize

生成 5 个训练可视化图表：

| 图表 | 文件名 | 内容 |
|------|--------|------|
| 损失曲线 | loss_curve.png | Train/Valid Loss 随 Epoch 变化，标注最佳 Epoch |
| 指标曲线 | metrics_curve.png | MSE/RMSE/PSNR/SSIM 四指标变化（2x2 子图） |
| 学习率曲线 | lr_curve.png | 学习率随 Epoch 变化 |
| 指标对比 | metrics_comparison.png | 验证集 vs 测试集指标柱状图 |
| 训练总结 | training_summary.png | 模型、参数、时长、最终指标表格 |

### 调用方式

```
ocean_sr_visualize({
  log_dir: "/path/to/training_output"
})
```

### 输出目录

```
log_dir/plots/
├── loss_curve.png
├── metrics_curve.png
├── lr_curve.png
├── metrics_comparison.png
└── training_summary.png
```

---

## 报告生成工具 ocean_sr_generate_report

生成包含可视化图表的 Markdown 报告。

### 调用方式

```
ocean_sr_generate_report({
  log_dir: "/path/to/training_output"
})
```

**重要**：先运行 `ocean_sr_visualize` 生成图表，再运行 `ocean_sr_generate_report`，报告会自动嵌入图表。

---

## 报告结构

生成的报告包含 8 个章节：

1. **训练配置** - 模型结构、数据配置、超参数、硬件配置
2. **训练过程** - 时间线、训练曲线、验证集性能演进
3. **最终性能评估** - 验证集/测试集指标
4. **可视化结果** - 图表文件列表 + 嵌入的图表图片
5. **模型检查点** - 保存的模型文件
6. **训练分析** - AI 填充的分析内容
7. **计算性能** - 训练时长、效率
8. **总结** - 核心成就、关键数据

---

## Agent 补充分析

报告中包含 `<!-- AI_FILL: ... -->` 占位符，Agent 需要：

1. 读取生成的报告文件
2. 分析报告中的数据和图表
3. 编写专业的分析内容替换占位符：
   - 损失下降趋势分析
   - 验证集与测试集性能对比
   - 可视化图表解读
   - 训练稳定性评估
   - 模型性能分析
   - 核心成就总结
4. 保存最终报告

---

## 完整流程示例

```
1. 用户询问训练状态
   → ocean_sr_train_status({ process_id: "xxx" })
   → 返回 status="completed"

2. Agent 主动询问
   → "训练已完成，是否需要生成可视化图表和训练报告？"

3. 用户确认后，生成可视化
   → ocean_sr_visualize({ log_dir: "..." })
   → 生成 5 个图表到 plots/

4. 生成报告
   → ocean_sr_generate_report({ log_dir: "..." })
   → 生成 training_report.md（含嵌入图表）

5. Agent 读取报告，补充分析
   → 读取 training_report.md
   → 替换 <!-- AI_FILL: ... --> 占位符
   → 保存最终报告

6. 向用户展示结果
   → 报告路径、关键指标、分析摘要
```
