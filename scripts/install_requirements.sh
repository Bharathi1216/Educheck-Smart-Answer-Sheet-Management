#!/usr/bin/env bash
# Minimal installer for Unix-like shells
# Usage: from project root: ./scripts/install_requirements.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="$ROOT_DIR/requirements.txt"

echo "Upgrading pip..."
python3 -m pip install --upgrade pip || python -m pip install --upgrade pip

if [ ! -f "$REQ_FILE" ]; then
  echo "requirements.txt not found at $REQ_FILE"
  exit 1
fi

echo "Installing packages from $REQ_FILE..."
python3 -m pip install -r "$REQ_FILE" || python -m pip install -r "$REQ_FILE"

echo "Done. To use a virtual environment:"
echo "python3 -m venv venv"
echo "source venv/bin/activate"
