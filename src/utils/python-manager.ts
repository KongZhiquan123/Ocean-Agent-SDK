/**
 * python-manager.ts
 *
 * Description: 管理和查找系统中可能的 Python 可执行文件路径
 * Author: kongzhiquan
 * Time: 2026-02-01
 * Version: 1.0.0
 *
 */
import { existsSync, readdirSync, statSync } from 'fs'
import os from 'os'
import path from 'path'

// 扫描系统中可能的 Python 可执行文件路径
export function findPossiblePythonPaths(): string[] {
  const home = os.homedir()
  const pyenvRoot = process.env.PYENV_ROOT || path.join(home, '.pyenv')

  const candidates = [
    ...collectFromEnv(),
    ...collectPyenvVersions(pyenvRoot),
    ...collectCommonLocations(home, pyenvRoot),
  ]

  return dedupeAndFilterExisting(candidates)
}

// 返回第一个可用的 Python 路径，找不到则返回 undefined
export function findFirstPythonPath(): string | undefined {
  return findPossiblePythonPaths()[0]
}

function collectFromEnv(): Array<string | undefined> {
  const isWin = process.platform === 'win32'
  return [
    process.env.PYTHON,
    process.env.PYTHON3,
    joinIf(process.env.PYTHON_HOME, isWin ? 'python.exe' : 'bin', isWin ? undefined : 'python3'),
    joinIf(process.env.VIRTUAL_ENV, isWin ? 'Scripts' : 'bin', isWin ? 'python.exe' : 'python'),
    joinIf(process.env.CONDA_PREFIX, isWin ? 'python.exe' : 'bin', isWin ? undefined : 'python'),
  ]
}



function collectPyenvVersions(pyenvRoot: string): string[] {
  const versionsDir = path.join(pyenvRoot, 'versions')
  try {
    const dirs = readdirSync(versionsDir, { withFileTypes: true })
    return dirs
      .filter((d) => d.isDirectory())
      .map((d) => path.join(versionsDir, d.name, 'bin', 'python'))
  } catch {
    return []
  }
}

function collectCommonLocations(home: string, pyenvRoot: string): Array<string | undefined> {
  const isWin = process.platform === 'win32'

  if (isWin) {
    const programFiles = process.env['ProgramFiles']
    const programFilesX86 = process.env['ProgramFiles(x86)']
    const paths: Array<string | undefined> = []

    paths.push(
      joinIf(home, 'anaconda3', 'python.exe'),
      joinIf(programFiles, 'Anaconda3', 'python.exe'),
      joinIf(programFilesX86, 'Anaconda3', 'python.exe'),
      'C:\\ProgramData\\Anaconda3\\python.exe',
    )

    return paths
  }

  const paths: Array<string | undefined> = [
    '/usr/bin/python3',
    '/usr/local/bin/python3',
    '/opt/homebrew/bin/python3',
    '/opt/local/bin/python3',
    '/usr/bin/python',
    '/usr/local/bin/python',
    joinIf(pyenvRoot, 'shims', 'python'),
  ]

  return paths
}

function dedupeAndFilterExisting(paths: Array<string | undefined>): string[] {
  const seen = new Set<string>()
  const results: string[] = []
  const isWin = process.platform === 'win32'

  for (const raw of paths) {
    if (!raw) continue
    const candidate = raw.trim()
    if (!candidate) continue

    const key = isWin ? candidate.toLowerCase() : candidate
    if (seen.has(key)) continue

    if (existsSync(candidate) && isExecutable(candidate)) {
      seen.add(key)
      results.push(candidate)
    }
  }

  return results
}


function isExecutable(target: string): boolean {
  try {
    const stat = statSync(target)
    return stat.isFile()
  } catch {
    return false
  }
}

function joinIf(base: string | undefined, ...parts: Array<string | undefined>): string | undefined {
  if (!base) return undefined
  const filtered = parts.filter((p): p is string => Boolean(p))
  return path.join(base, ...filtered)
}
