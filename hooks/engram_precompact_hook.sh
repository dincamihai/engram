#!/bin/bash
# Engram pre-compact hook — remind Claude to save important context before compaction.
# Called by Claude Code's PreCompact hook.

echo '{"decision":"approve","systemMessage":"Context is about to be compacted. Before proceeding, review the conversation for any important information worth saving to engram memory (engram_store): decisions made, issues resolved, patterns discovered, deployment outcomes. Save anything that would be useful in future sessions."}'
