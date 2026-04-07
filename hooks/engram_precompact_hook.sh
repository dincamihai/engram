#!/bin/bash
# Engram pre-compact hook — save context before Claude Code compresses conversation.
# Called by Claude Code's PreCompact hook.

ENGRAM="/Users/mid/Repos/engram/target/release/engram"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-embeddinggemma}"

# Read the transcript summary from stdin (Claude Code pipes it)
SUMMARY=$(cat)

if [ -z "$SUMMARY" ]; then
    exit 0
fi

$ENGRAM store "$SUMMARY" --source "claude-code-precompact" 2>/dev/null

exit 0
