#!/usr/bin/env bash
# CI test script — runs fmt, clippy, tests, and no_std checks.
#
# Usage:
#   ./scripts/ci-test.sh
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more checks failed

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0
STEPS=()

run_step() {
    local name="$1"
    shift
    echo ""
    echo "==> [$name]"
    if "$@" 2>&1; then
        echo "  ✓ PASS: $name"
        PASS=$((PASS + 1))
        STEPS+=("PASS: $name")
    else
        echo "  ✗ FAIL: $name"
        FAIL=$((FAIL + 1))
        STEPS+=("FAIL: $name")
    fi
}

# 1. Format check
run_step "cargo fmt --check" cargo fmt --check

# 2. Clippy (exclude rustyhdf5-py which needs PyO3/Python)
run_step "cargo clippy" cargo clippy \
    --workspace \
    --exclude rustyhdf5-py \
    -- -D warnings

# 3. Tests (exclude rustyhdf5-py)
run_step "cargo test" cargo test \
    --workspace \
    --exclude rustyhdf5-py

# 4. no_std check
run_step "check-nostd.sh" "$SCRIPT_DIR/check-nostd.sh"

# Summary
echo ""
echo "========================================"
echo "  CI Summary"
echo "========================================"
for s in "${STEPS[@]}"; do
    echo "  $s"
done
echo "----------------------------------------"
echo "  $PASS passed, $FAIL failed"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
