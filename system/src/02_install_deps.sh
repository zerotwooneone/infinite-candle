#!/bin/bash
set -e

# 1. Enforce Root
if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run as root (sudo bash 02_sys_deps.sh)"
  exit
fi

echo "🛠️  Installing System Build Tools..."

# Install standard compilation tools required for Python C-Extensions
apt-get install -y \
    build-essential \
    python3-dev \
    git \
    scons \
    swig \
    unzip \
    wget \
    curl

echo "✅ System Dependencies Installed."