/**
 * @file python-manager.ts
 *
 * @description 管理和查找系统中可能的 Python 可执行文件路径（异步实现）
 * @author kongzhiquan
 * @contributors Leizheng
 * @date 2026-02-01
 * @version 1.2.0
 *
 * @changelog
 *   - 2026-03-05 kongzhiquan: v1.2.0
 *     - 将 Python 路径扫描流程改为异步实现
 *     - 将模块探测从 execSync 改为异步子进程调用，避免阻塞事件循环
 *   - 2026-02-07 Leizheng: v1.1.0
 *     - 扫描当前用户的 conda 环境 (miniconda3, anaconda3)
 *     - 新增 findPythonWithModule(moduleName) 查找包含指定模块的 Python
 */
import { constants as fsConstants } from 'fs'
import { access, readdir, stat } from 'fs/promises'
import { execFile } from 'child_process'
import { promisify } from 'util'
import os from 'os'
import path from 'path'

const execFileAsync = promisify(execFile)

// 扫描系统中可能的 Python 可执行文件路径
export async function findPossiblePythonPaths(): Promise<string[]> {
  const home = os.homedir()
  const pyenvRoot = process.env.PYENV_ROOT || path.join(home, '.pyenv')

  const pyenvPaths = await collectPyenvVersions(pyenvRoot)
  const commonPaths = await collectCommonLocations(home, pyenvRoot)
  const candidates = [
    ...collectFromEnv(),
    ...pyenvPaths,
    ...commonPaths,
  ]

  return await dedupeAndFilterExisting(candidates)
}

// 返回第一个可用的 Python 路径，找不到则返回 undefined
export async function findFirstPythonPath(): Promise<string | undefined> {
  const paths = await findPossiblePythonPaths()
  return paths[0]
}

// 查找包含指定模块的 Python 路径（如 findPythonWithModule('torch')）
// 会缓存结果避免重复检测
const _moduleCache = new Map<string, string | undefined>()
/** 合法的 Python 模块名格式（防止命令注入） */
const MODULE_NAME_PATTERN = /^[a-zA-Z_][a-zA-Z0-9_.]*$/
export async function findPythonWithModule(moduleName: string): Promise<string | undefined> {
  if (!MODULE_NAME_PATTERN.test(moduleName)) {
    console.error(`[python-manager] Invalid module name: "${moduleName}"`)
    return undefined
  }
  if (_moduleCache.has(moduleName)) {
    return _moduleCache.get(moduleName)
  }
  const allPaths = await findPossiblePythonPaths()
  for (const pyPath of allPaths) {
    try {
      await execFileAsync(pyPath, ['-c', `import ${moduleName}`], {
        timeout: 10000,
        windowsHide: true,
      })
      _moduleCache.set(moduleName, pyPath)
      return pyPath
    } catch {
      // 该 Python 没有此模块，继续找
    }
  }
  _moduleCache.set(moduleName, undefined)
  return undefined
}

function collectFromEnv(): Array<string | undefined> {
  const isWin = process.platform === 'win32'
  return [
    process.env.PYTHON3,
    process.env.PYTHON,
    joinIf(process.env.PYTHON_HOME, isWin ? 'python.exe' : 'bin', isWin ? undefined : 'python3'),
    joinIf(process.env.VIRTUAL_ENV, isWin ? 'Scripts' : 'bin', isWin ? 'python.exe' : 'python'),
    joinIf(process.env.CONDA_PREFIX, isWin ? 'python.exe' : 'bin', isWin ? undefined : 'python'),
  ]
}

async function collectPyenvVersions(pyenvRoot: string): Promise<string[]> {
  const versionsDir = path.join(pyenvRoot, 'versions')
  try {
    const dirs = await readdir(versionsDir, { withFileTypes: true })
    return dirs
      .filter((d) => d.isDirectory())
      .map((d) => path.join(versionsDir, d.name, 'bin', 'python'))
  } catch {
    return []
  }
}

async function collectCommonLocations(home: string, pyenvRoot: string): Promise<Array<string | undefined>> {
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
    ...(await collectCondaEnvs()),
  ]

  return paths
}

// 扫描当前用户的 conda 环境 (miniconda3/envs, anaconda3/envs)
async function collectCondaEnvs(): Promise<string[]> {
  const home = os.homedir()
  const results: string[] = []
  for (const condaDir of ['miniconda3', 'anaconda3']) {
    const envsDir = path.join(home, condaDir, 'envs')
    try {
      const envs = await readdir(envsDir, { withFileTypes: true })
      for (const env of envs) {
        if (env.isDirectory()) {
          results.push(path.join(envsDir, env.name, 'bin', 'python'))
        }
      }
    } catch {
      // 目录不存在或不可读
    }
  }
  return results
}

async function dedupeAndFilterExisting(paths: Array<string | undefined>): Promise<string[]> {
  const seen = new Set<string>()
  const results: string[] = []
  const isWin = process.platform === 'win32'

  for (const raw of paths) {
    if (!raw) continue
    const candidate = raw.trim()
    if (!candidate) continue

    const key = isWin ? candidate.toLowerCase() : candidate
    if (seen.has(key)) continue

    if (await pathExists(candidate) && await isExecutable(candidate)) {
      seen.add(key)
      results.push(candidate)
    }
  }

  return results
}


async function isExecutable(target: string): Promise<boolean> {
  try {
    const statResult = await stat(target)
    if (!statResult.isFile()) {
      return false
    }
    if (process.platform === 'win32') {
      return true
    }
    await access(target, fsConstants.X_OK)
    return true
  } catch {
    return false
  }
}

async function pathExists(target: string): Promise<boolean> {
  try {
    await access(target, fsConstants.F_OK)
    return true
  } catch {
    return false
  }
}

function joinIf(base: string | undefined, ...parts: Array<string | undefined>): string | undefined {
  if (!base) return undefined
  const filtered = parts.filter((p): p is string => Boolean(p))
  return path.join(base, ...filtered)
}
