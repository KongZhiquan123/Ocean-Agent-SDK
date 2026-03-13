/**
 * @file validate.ts
 * @description Forecast 模块 validate 入口 — 透传导出 SR 共用工具
 *              ocean_validate_tensor 工具为 SR/Forecast 共用，已通过 SR 模块注册到全局
 *              此文件仅做具名导出，供模块消费者直接引用，不新增工具注册
 *
 * @author kongzhiquan
 * @date 2026-02-26
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-26 kongzhiquan: v1.0.0 初始版本，re-export SR validate 工具
 */

export { oceanValidateTensorTool } from '../ocean-SR-data-preprocess'