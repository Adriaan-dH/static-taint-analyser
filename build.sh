#!/usr/bin/env bash

set -eu

command -v python3 >/dev/null 2>&1 || {
    echo "build.sh: python3 is required" >&2
    exit 1
}
python3 -B -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "build.sh: Python 3.10 or later is required")'
chmod +x main
