/**
 * @file shell.ts
 *
 * @description Shell 命令构建相关的工具函数
 * @author kongzhiquan
 * @date 2026-03-04
 * @version 1.1.0
 *
 * @changelog
 *   - 2026-03-04 kongzhiquan: v1.1.0 新增 extractTaggedJson，从训练工具中提取公共逻辑
 *   - 2026-03-04 kongzhiquan: 初始版本
 */

// ========================================
// 转义命令 - 避免shell注入
// ========================================

/**
 * 转义字符串中的特殊 shell 字符，使其可安全用于双引号包裹的参数。
 * 转义范围：\ " $ ` !
 */
export function shellEscapeDouble(str: string): string {
  return str.replace(/[\\"$`!]/g, '\\$&')
}

/**
 * 将 JSON 字符串转义，使其可安全嵌入 shell 单引号参数中。
 * 单引号内无法转义，因此将 ' 替换为 '\''。
 */
export function shellSafeJson(json: string): string {
  return json.replace(/'/g, "'\\''")
}

// ========================================
// 权限控制 - 危险命令黑名单
// ========================================
export const DANGEROUS_PATTERNS = [
  /rm\s+(-[rRf]+\s+)*[\/~]/,         // rm -rf / 或 rm -rf ~
  /rm\s+(-[rRf]+\s+)*\.\./,          // rm -rf ..
  />\s*\/etc\//,                      // 重定向写入 /etc/
  />\s*\/usr\//,                      // 重定向写入 /usr/
  />\s*\/bin\//,                      // 重定向写入 /bin/
  /sudo\s+/,                          // sudo 命令
  /chmod\s+777/,                      // 危险权限
  /chown\s+root/,                     // 改变所有者为 root
  /mkfs/,                             // 格式化磁盘
  /dd\s+.*of=\/dev/,                  // 写入设备
  /:(){ :|:& };:/,                    // fork 炸弹
  />\s*\/dev\/(sda|hda|nvme)/,        // 写入磁盘设备
  /curl.*\|\s*(ba)?sh/,               // curl | bash 远程执行
  /wget.*\|\s*(ba)?sh/,               // wget | bash 远程执行
] as const

export const SHELL_CONTROL_PATTERN = /[;&`<>]/ // 禁止命令拼接/重定向
export const SUBSHELL_PATTERN = /\$\(/ // 禁止 $() 子命令
export const DOLLAR_PATTERN = /\$/ // 禁止环境变量展开
export const NEWLINE_PATTERN = /[\r\n]/ // 禁止换行注入

export const SAFE_READ_PATTERNS = [
  /^pwd$/,
  /^whoami$/,
  /^id$/,
  /^date$/,
  /^ls(\s+[-\w./]+)*$/,
  /^cat\s+[-\w./]+$/,
  /^head(\s+-n\s+\d+)?\s+[-\w./]+$/,
  /^tail(\s+-n\s+\d+)?\s+[-\w./]+$/,
  /^sed\s+-n\s+['"]?\d+(,\d+)?p['"]?\s+[-\w./]+$/,
  /^rg\s+['"][^'"]+['"](\s+[-\w./]+)*$/,
  /^(grep|egrep|fgrep)(\s+-[a-zA-Z]+)*\s+[^|]+\s+[-\w./]+$/,
  /^diff(\s+-[a-zA-Z]+)*\s+[-\w./]+\s+[-\w./]+$/,
  /^tree(\s+-[a-zA-Z]+)*(\s+[-\w./]+)?$/,
  /^wc(\s+-[clmw]+)?\s+[-\w./]+$/,
  /^stat\s+[-\w./]+$/,
  /^du(\s+-h)?\s+[-\w./]+$/,
  /^df(\s+-h)?$/,
] as const

export const SAFE_WRITE_PATTERNS = [
  /^mkdir(\s+-p)?\s+[-\w./]+$/,
  /^touch\s+[-\w./]+$/,
  /^cp(\s+-[a-zA-Z]+)?\s+[-\w./]+\s+[-\w./]+$/,
  /^mv(\s+-[a-zA-Z]+)?\s+[-\w./]+\s+[-\w./]+$/,
] as const

export function isDangerousCommand(command: string): boolean {
  return DANGEROUS_PATTERNS.some(pattern => pattern.test(command))
}

export function hasShellControlChars(command: string): boolean {
  return (
    SHELL_CONTROL_PATTERN.test(command) ||
    SUBSHELL_PATTERN.test(command) ||
    DOLLAR_PATTERN.test(command) ||
    NEWLINE_PATTERN.test(command)
  )
}

export function isWhitelisted(command: string, patterns: RegExp[] | readonly RegExp[]): boolean {
  return patterns.some(pattern => pattern.test(command))
}

export function splitPipeline(command: string): string[] {
  return command.split('|').map((part) => part.trim()).filter(Boolean)
}

export function isReadOnlyPipeline(command: string): boolean {
  if (!command.includes('|')) return false
  const parts = splitPipeline(command)
  if (parts.length === 0) return false
  return parts.every(part => isWhitelisted(part, SAFE_READ_PATTERNS))
}

export function isAllowedPathToken(token: string, allowedPaths?: string[]): boolean {
  const trimmed = token.replace(/^['"]|['"]$/g, '')
  if (!trimmed || trimmed.startsWith('-')) return true
  if (allowedPaths && allowedPaths.some(path => trimmed.startsWith(path))) return true
  if (trimmed.includes('..')) return false
  if (trimmed.startsWith('~')) return false
  return true
}

export function hasUnsafePath(command: string, allowedPaths?: string[]): boolean {
  const tokens = command.split(/\s+/)
  return tokens.some(token => !isAllowedPathToken(token, allowedPaths))
}

// ========================================
// Python 脚本输出解析
// ========================================

/**
 * 从 Python 脚本的 stdout 中提取 __TAG__....__TAG__ 格式包裹的 JSON 数据。
 * 解析失败或未找到时返回 null。
 */
export function extractTaggedJson(output: string, tag: string): Record<string, unknown> | null {
  const pattern = new RegExp(`__${tag}__([\\s\\S]*?)__${tag}__`)
  const match = output.match(pattern)
  if (!match) return null
  try {
    return JSON.parse(match[1])
  } catch {
    return null
  }
}
