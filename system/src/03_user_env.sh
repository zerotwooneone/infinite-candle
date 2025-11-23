#!/bin/bash
set -e

# 1. Enforce NON-Root
if [ "$EUID" -eq 0 ]; then
  echo "❌ Do NOT run this as root. Run as the normal user (pi)."
  exit
fi

echo "⚡ Setting up User Environment..."

# 2. Install uv (if not found)
if ! command -v uv &> /dev/null; then
    echo "   -> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source cargo env to make uv available immediately in this script
    source "$HOME/.cargo/env"
else
    echo "   -> uv is already installed."
fi

# 3. Navigate to Application Directory
# We assume this script is running from infinite-candle/system/src/
# So we go up two levels, then into lights-control
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../lights-control" && pwd)"

if [ -d "$PROJECT_DIR" ]; then
    echo "📂 Found project at: $PROJECT_DIR"
    cd "$PROJECT_DIR"

    # 4. Sync Dependencies
    echo "🔄 Running uv sync..."
    uv sync
    
    echo "✅ Environment Ready."
    echo "   -> You can now run the lights with: sudo .venv/bin/python src/main.py"
else
    echo "❌ Could not find 'lights-control' directory. Are you running this from inside 'system/src'?"
    exit 1
fi