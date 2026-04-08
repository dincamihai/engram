#!/bin/bash
# Engram session start hook — inject recent memories into context.
# Called by Claude Code's SessionStart hook.

ENGRAM="${ENGRAM_HOME:-$HOME/.engram}/bin/engram"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-embeddinggemma}"

# Use current date to bias semantic search toward recent memories
TODAY=$(date +%Y-%m-%d)
MONTH=$(date +"%B %Y")
QUERY="$MONTH recent work context $TODAY"

RESULT=$("$ENGRAM" search "$QUERY" --limit 10 2>/dev/null | head -100)

if [ -n "$RESULT" ]; then
  ESCAPED=$(echo "$RESULT" | python3 -c "import sys,json; s=json.dumps(sys.stdin.read()); print(s[1:-1])")
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":\"== Engram Memory (auto-loaded) ==\n${ESCAPED}\n== End Engram Memory ==\"}}"
fi
