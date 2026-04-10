#!/bin/bash
# Engram pre-compact hook — store important context before conversation compaction.
# The agent is unaware — memories are saved automatically.

ENGRAM="${ENGRAM_HOME:-$HOME/.engram}/bin/engram"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-embeddinggemma}"

# Read the transcript summary from stdin (Claude Code pipes it)
SUMMARY=$(cat)

if [ -n "$SUMMARY" ]; then
    # Store the session summary in background
    "$ENGRAM" store "$SUMMARY" --source "claude-code-compact" 2>/dev/null &
fi

# Approve the compaction
echo '{"decision":"approve"}'

exit 0