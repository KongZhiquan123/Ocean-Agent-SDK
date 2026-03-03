#!/usr/bin/env bash
# PreToolUse hook: block dangerous bash commands
# Exit code 2 = block the command
# Exit code 0 = allow the command

set -euo pipefail

INPUT=$(cat)

CMD=$(node -e "
  let d='';
  process.stdin.on('data',c=>d+=c);
  process.stdin.on('end',()=>{
    try {
      const o=JSON.parse(d);
      console.log(o.tool_input?.command || '');
    } catch(e) { console.log(''); }
  });
" <<< "$INPUT")

if [ -z "$CMD" ]; then
  echo "$INPUT"
  exit 0
fi

# Dangerous patterns - block with exit code 2
BLOCKED=false
REASON=""

# Destructive file operations
if echo "$CMD" | grep -qE 'rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?(/\s|/\*|~|/home\b)'; then
  BLOCKED=true
  REASON="Destructive rm on system/home directory"
fi

# Sudo
if echo "$CMD" | grep -qE '^\s*sudo\b'; then
  BLOCKED=true
  REASON="sudo command"
fi

# System directory writes
if echo "$CMD" | grep -qE '>\s*/(etc|usr|bin|sbin|boot)/'; then
  BLOCKED=true
  REASON="Write to system directory"
fi

# Disk operations
if echo "$CMD" | grep -qE '\b(mkfs|dd\s+if=|fdisk|parted)\b'; then
  BLOCKED=true
  REASON="Disk operation"
fi

# Fork bombs
if echo "$CMD" | grep -qE ':\(\)\{.*\|.*\}'; then
  BLOCKED=true
  REASON="Fork bomb"
fi

# Remote code execution
if echo "$CMD" | grep -qE 'curl.*\|\s*(ba)?sh|wget.*\|\s*(ba)?sh'; then
  BLOCKED=true
  REASON="Remote code execution (curl|bash)"
fi

# Leaking secrets
if echo "$CMD" | grep -qE '\b(cat|less|more|head|tail)\s+.*\.env\b|printenv|env\s*$'; then
  BLOCKED=true
  REASON="Potential secret exposure"
fi

if [ "$BLOCKED" = true ]; then
  echo "[hook] BLOCKED: $REASON" >&2
  echo "[hook] Command: $CMD" >&2
  exit 2
fi

echo "$INPUT"
