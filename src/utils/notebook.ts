/**
 * @file notebook.ts
 *
 * @description Jupyter Notebook 通用工具
 *              提供 NotebookCell / Notebook 类型、辅助函数与通用的保存/追加逻辑
 *              供 ocean-preprocess 和 ocean-sr-training 等工具共享
 *
 * @author kongzhiquan
 * @date 2026-02-25
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-25 kongzhiquan: v1.0.0 从 ocean-preprocess/notebook.ts 提取公共逻辑
 */

import type { EnhancedToolContext } from '@shareai-lab/kode-sdk'

// ========================================
// 类型定义
// ========================================

export interface NotebookCell {
  cell_type: 'code' | 'markdown'
  metadata: Record<string, unknown>
  source: string[]
  outputs?: unknown[]
  execution_count?: number | null
}

export interface Notebook {
  nbformat: number
  nbformat_minor: number
  metadata: Record<string, unknown>
  cells: NotebookCell[]
}

// ========================================
// 辅助函数
// ========================================

/**
 * 将 JavaScript 值转为 Python 字面量表示
 */
export function toPyRepr(value: unknown): string {
  if (value === undefined || value === null) return 'None'
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return JSON.stringify(value)
  if (Array.isArray(value)) {
    const items = value.map(v => toPyRepr(v))
    return `[${items.join(', ')}]`
  }
  return JSON.stringify(value)
}

/**
 * 将多行字符串拆成 Notebook source 数组（非末行加 \n）
 */
function toSourceLines(source: string): string[] {
  return source.split('\n').map((line, i, arr) =>
    i < arr.length - 1 ? line + '\n' : line
  )
}

/**
 * 创建 markdown cell
 */
export function mdCell(source: string): NotebookCell {
  return {
    cell_type: 'markdown',
    metadata: {},
    source: toSourceLines(source),
  }
}

/**
 * 创建 code cell
 */
export function codeCell(source: string): NotebookCell {
  return {
    cell_type: 'code',
    metadata: {},
    source: toSourceLines(source),
    outputs: [],
    execution_count: null,
  }
}

/**
 * 创建空的 nbformat v4 Notebook
 */
export function createEmptyNotebook(): Notebook {
  return {
    nbformat: 4,
    nbformat_minor: 2,
    metadata: {
      kernelspec: {
        display_name: 'Python 3',
        language: 'python',
        name: 'python3',
      },
      language_info: {
        name: 'python',
        version: '3.10.0',
        mimetype: 'text/x-python',
        file_extension: '.py',
      },
    },
    cells: [],
  }
}

/**
 * 保存或追加 Notebook
 * 如果文件已存在，在末尾追加新的 cells；否则创建新 Notebook
 */
export async function saveOrAppendNotebook(
  ctx: EnhancedToolContext,
  notebookPath: string,
  newCells: NotebookCell[]
): Promise<void> {
  let notebook: Notebook

  try {
    const content = await ctx.sandbox.fs.read(notebookPath)
    const parsed = JSON.parse(content)

    if (parsed.nbformat && Array.isArray(parsed.cells)) {
      notebook = parsed as Notebook

      // 追加分隔 cell
      const timestamp = new Date().toISOString().replace('T', ' ').slice(0, 19)
      notebook.cells.push(mdCell(
        `---\n\n## 记录追加于 ${timestamp}`
      ))
      notebook.cells.push(...newCells)
    } else {
      notebook = createEmptyNotebook()
      notebook.cells = newCells
    }
  } catch {
    // 文件不存在或解析失败，创建新 notebook
    notebook = createEmptyNotebook()
    notebook.cells = newCells
  }

  await ctx.sandbox.fs.write(notebookPath, JSON.stringify(notebook, null, 1))
}
