import { oceanInspectDataTool } from './inspect'
import { oceanValidateTensorTool } from './validate'
import { oceanConvertNpyTool } from './convert'
import { oceanPreprocessFullTool } from './full'

export const oceanPreprocessTools = [
  oceanInspectDataTool,
  oceanValidateTensorTool,
  oceanConvertNpyTool,
  oceanPreprocessFullTool
]

export {
  oceanInspectDataTool,
  oceanValidateTensorTool,
  oceanConvertNpyTool,
  oceanPreprocessFullTool
}

export default oceanPreprocessTools
