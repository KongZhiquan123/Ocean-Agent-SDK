#!/usr/bin/env bash
# PostToolUse hook: auto-verify after Edit/Write
# - TS/TSX files → npm run typecheck
# - Python files → python -m py_compile
#
# Reads JSON from stdin, checks file_path extension, runs verification.
# Outputs the original JSON unchanged. Verification results go to stderr.

set -euo pipefail

INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | node -e "
  let d='';
  process.stdin.on('data',c=>d+=c);
  process.stdin.on('end',()=>{
    try {
      const o=JSON.parse(d);
      console.log(o.tool_input?.file_path || '');
    } catch(e) { console.log(''); }
  });
" 2>/dev/null <<< "$INPUT")

if [ -z "$FILE_PATH" ]; then
  echo "$INPUT"
  exit 0
fi

# Get file extension
EXT="${FILE_PATH##*.}"

case "$EXT" in
  ts|tsx)
    echo "[hook] Running typecheck after editing $FILE_PATH ..." >&2
    cd /home/lz/Ocean-Agent-SDK
    if npm run typecheck 2>&1 | tail -5 >&2; then
      echo "[hook] Typecheck passed." >&2
    else
      echo "[hook] WARNING: Typecheck failed! Review the errors above." >&2
    fi
    ;;
  py)
    echo "[hook] Running py_compile on $FILE_PATH ..." >&2
    if /home/lz/miniconda3/envs/pytorch/bin/python -m py_compile "$FILE_PATH" 2>&1 >&2; then
      echo "[hook] py_compile passed." >&2
    else
      echo "[hook] WARNING: py_compile failed! Syntax error in $FILE_PATH" >&2
    fi
    ;;
esac

# Always pass through the original input
echo "$INPUT"
