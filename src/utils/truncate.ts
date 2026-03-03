/**
 * @file notebook.ts
 *
 * @description 数据截断工具，用于对特殊对象（例如Dict, Map）进行截断处理，避免过长的输出导致问题
 *
 * @author kongzhiquan
 * @date 2026-02-25
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-25 kongzhiquan: 初始版本
 */
const truncateDict = (input: Record<string, any>, max = 20, note = '数据过长而被截断') => {
  if (!input) return input
  const entries = Object.entries(input)
  if (entries.length <= max) return input

  const truncatedEntries = entries.slice(0, max)
  return {
    _truncated: true,
    _total: entries.length,
    _note: note,
    ...Object.fromEntries(truncatedEntries)
  }
}

const truncateMap = (input: Map<any, any>, max = 20, note = '数据过长而被截断') => {
  if (!input) return input
  const entries = Array.from(input.entries())
  if (entries.length <= max) return input

  const truncatedEntries = entries.slice(0, max)
  return {
    _truncated: true,
    _total: entries.length,
    _note: note,
    ...Object.fromEntries(truncatedEntries)
  }
}

export {
    truncateDict,
    truncateMap
}