/**
 * @file inspect.ts
 * @description Forecast 模块 inspect 入口 — 透传导出 SR 共用工具
 *              ocean_inspect_data 工具为 SR/Forecast 共用，已通过 SR 模块注册到全局
 *              此文件仅做具名导出，供模块消费者直接引用，不新增工具注册
 *
 * @author Leizheng
 * @date 2026-02-26
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-26 Leizheng: v1.0.0 初始版本，re-export SR inspect 工具
 */

export { oceanInspectDataTool } from '../ocean-SR-data-preprocess/inspect'
