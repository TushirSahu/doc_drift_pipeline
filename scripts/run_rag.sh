#!/usr/bin/env bash

set -euo pipefail

QUESTION="${1:-What does the auth service use for authentication?}"
python -m src.agentic.cli "$QUESTION"
