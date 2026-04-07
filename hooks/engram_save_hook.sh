#!/bin/bash
# Engram stop hook — auto-save session context on conversation stop.
# Called by Claude Code's Stop hook.

ENGRAM="/Users/mid/Repos/engram/target/release/engram"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-embeddinggemma}"

# Read the transcript summary from stdin (Claude Code pipes it)
SUMMARY=$(cat)

if [ -z "$SUMMARY" ]; then
    exit 0
fi

# Store the session summary
$ENGRAM store "$SUMMARY" --source "claude-code-session" 2>/dev/null

exit 0
