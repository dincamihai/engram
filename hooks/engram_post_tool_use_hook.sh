#!/bin/bash
# Engram post-tool-use hook — store noteworthy results AND activate relevant context.
# The brain both learns (stores) and recalls (injects) as the agent works.

ENGRAM="${ENGRAM_HOME:-$HOME/.engram}/bin/engram"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-embeddinggemma}"

# Read tool use data from stdin (JSON)
INPUT=$(cat)

# Extract the tool name
TOOL=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_name',''))" 2>/dev/null)

# Extract result (truncated to avoid storing huge outputs)
RESULT=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',''); print(r[:2000] if isinstance(r,str) else str(r)[:2000])" 2>/dev/null)

ACTIVATED=""

case "$TOOL" in
    Bash|Agent)
        # These produce noteworthy results — store in background
        if [ -n "$RESULT" ]; then
            "$ENGRAM" store "$RESULT" --source "claude-code-hook" 2>/dev/null &

            # Activate relevant memories based on the result
            CONTEXT=$("$ENGRAM" search "$RESULT" --limit 5 2>/dev/null | head -60)
            if [ -n "$CONTEXT" ]; then
                ACTIVATED=$(echo "$CONTEXT" | python3 -c "
import sys, json
try:
    lines = sys.stdin.read().strip()
    if not lines:
        sys.exit(0)
    escaped = json.dumps(lines)[1:-1]
    print(escaped)
except:
    sys.exit(0)
" 2>/dev/null)
            fi
        fi
        ;;
esac

# Inject activated context if we found anything
if [ -n "$ACTIVATED" ]; then
    echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PostToolUse\",\"additionalContext\":\"== Activated Memory ==\n${ACTIVATED}\n== End Activated Memory ==\"}}"
fi

exit 0