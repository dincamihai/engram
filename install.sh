#!/bin/bash
set -euo pipefail

REPO="dincamihai/engram"
INSTALL_DIR="${ENGRAM_HOME:-$HOME/.engram}"
ENGRAM_BIN="$INSTALL_DIR/bin/engram"
HOOKS_DIR="$INSTALL_DIR/hooks"
SETTINGS="$HOME/.claude/settings.local.json"

# Detect platform
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$OS" in
  linux)  OS="linux" ;;
  darwin) OS="darwin" ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac
case "$ARCH" in
  x86_64|amd64)  ARCH="amd64" ;;
  aarch64|arm64) ARCH="arm64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac
ARTIFACT="engram-${OS}-${ARCH}"

# Determine version
VERSION="${1:-latest}"

echo "==> Downloading engram ($ARTIFACT)..."
mkdir -p "$INSTALL_DIR/bin" "$INSTALL_DIR/hooks"

if [ "$VERSION" = "latest" ]; then
  DOWNLOAD_URL="https://github.com/$REPO/releases/latest/download/$ARTIFACT"
else
  DOWNLOAD_URL="https://github.com/$REPO/releases/download/$VERSION/$ARTIFACT"
fi

curl -fSL "$DOWNLOAD_URL" -o "$ENGRAM_BIN"
chmod +x "$ENGRAM_BIN"

echo "==> Installing hooks..."
# Download hooks from the repo (use the tag if specified, otherwise main)
REF="${VERSION:-main}"
if [ "$REF" = "latest" ]; then REF="main"; fi
for hook in engram_save_hook.sh engram_precompact_hook.sh engram_session_start_hook.sh; do
  curl -fSL "https://raw.githubusercontent.com/$REPO/$REF/hooks/$hook" -o "$HOOKS_DIR/$hook"
  chmod +x "$HOOKS_DIR/$hook"
  # Patch hook to use installed binary path
  sed -i.bak "s|ENGRAM=.*|ENGRAM=\"$ENGRAM_BIN\"|" "$HOOKS_DIR/$hook"
  rm -f "$HOOKS_DIR/$hook.bak"
done

echo "==> Registering MCP server..."
claude mcp remove memory --scope user 2>/dev/null || true
claude mcp add --scope user memory \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  -e OLLAMA_EMBED_MODEL=embeddinggemma \
  -- "$ENGRAM_BIN" serve

echo "==> Configuring hooks and permissions..."
mkdir -p "$HOME/.claude"

python3 - "$SETTINGS" "$HOOKS_DIR" <<'PYEOF'
import json, sys, os

settings_path = sys.argv[1]
hooks_dir = sys.argv[2]

if os.path.exists(settings_path):
    with open(settings_path) as f:
        settings = json.load(f)
else:
    settings = {}

# Permissions
perms = settings.setdefault("permissions", {})
allow = perms.setdefault("allow", [])
for tool in ["mcp__memory__engram_store", "mcp__memory__engram_search"]:
    if tool not in allow:
        allow.append(tool)

# Hooks
hooks = settings.setdefault("hooks", {})
hooks["Stop"] = [{"hooks": [{"type": "command", "command": f"{hooks_dir}/engram_save_hook.sh"}]}]
hooks["PreCompact"] = [{"hooks": [{"type": "command", "command": f"{hooks_dir}/engram_precompact_hook.sh"}]}]
hooks["SessionStart"] = [{"hooks": [{"type": "command", "command": f"{hooks_dir}/engram_session_start_hook.sh", "timeout": 15, "statusMessage": "Loading memories..."}]}]

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")
PYEOF

echo ""
echo "Done! engram installed to $INSTALL_DIR"
echo ""
echo "Prerequisites:"
echo "  ollama pull embeddinggemma"
