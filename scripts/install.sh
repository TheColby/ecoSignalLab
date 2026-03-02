#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: bash scripts/install.sh [--python PYTHON] [--venv DIR] [--extras EXTRAS] [--system] [--non-editable]

Installs ecoSignalLab and ensures man pages are available under <prefix>/share/man/man1.

options:
  --python PYTHON      Python executable (default: python3)
  --venv DIR           Virtualenv directory (default: .venv)
  --extras EXTRAS      Optional extras list (default: dev,ml,plot,io,docs,features)
  --system             Install into current interpreter environment (no venv creation)
  --non-editable       Use standard install instead of editable install
  -h, --help           Show this help
EOF
}

PYTHON_BIN="python3"
VENV_DIR=".venv"
EXTRAS="dev,ml,plot,io,docs,features"
USE_SYSTEM=0
EDITABLE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --extras)
      EXTRAS="$2"
      shift 2
      ;;
    --system)
      USE_SYSTEM=1
      shift
      ;;
    --non-editable)
      EDITABLE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$USE_SYSTEM" -eq 0 ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  PYTHON_EXEC="$VENV_DIR/bin/python"
else
  PYTHON_EXEC="$PYTHON_BIN"
fi

"$PYTHON_EXEC" -m pip install --upgrade pip

TARGET="."
if [[ -n "$EXTRAS" ]]; then
  TARGET=".[${EXTRAS}]"
fi
if [[ "$EDITABLE" -eq 1 ]]; then
  "$PYTHON_EXEC" -m pip install -e "$TARGET"
else
  "$PYTHON_EXEC" -m pip install "$TARGET"
fi

PREFIX="$("$PYTHON_EXEC" - <<'PY'
import sys
print(sys.prefix)
PY
)"

MAN_DIR="${PREFIX}/share/man/man1"
mkdir -p "$MAN_DIR"
cp man/man1/*.1 "$MAN_DIR/"

echo "installed prefix: $PREFIX"
echo "man pages installed to: $MAN_DIR"
echo "try:"
echo "  MANPATH=\"$PREFIX/share/man:\$MANPATH\" man esl"
echo "  MANPATH=\"$PREFIX/share/man:\$MANPATH\" man esl-analyze"
