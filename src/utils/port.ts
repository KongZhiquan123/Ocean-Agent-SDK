/**
 * @file port.ts
 *
 * @description 网络端口工具函数：检测端口占用、查找空闲端口
 * @author kongzhiquan
 * @date 2026-03-04
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-04 kongzhiquan: 初始版本，从 ocean-SR-training/train.ts 和 ocean-forecast-training/train.ts 提取公共逻辑
 */

import net from 'node:net'

/**
 * 检测指定端口是否空闲（未被占用）。
 */
export async function isPortFree(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer()
    server.unref()
    server.once('error', () => resolve(false))
    server.listen(port, () => {
      server.close(() => resolve(true))
    })
  })
}

/**
 * 在指定范围内查找第一个空闲端口，找不到则返回 null。
 */
export async function findFreePort(start = 29500, end = 29600): Promise<number | null> {
  for (let port = start; port <= end; port += 1) {
    // eslint-disable-next-line no-await-in-loop
    if (await isPortFree(port)) return port
  }
  return null
}
