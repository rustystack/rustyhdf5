#!/usr/bin/env bash
# Cross-compilation validation for rustyhdf5-format.
#
# Verifies the library builds for multiple targets:
#   1. no_std (ARM Cortex-M) — core target
#   2. aarch64-unknown-linux-gnu (Android NDK style)
#   3. wasm32-unknown-unknown
#   4. x86_64-unknown-linux-gnu
#
# Usage:
#   ./scripts/cross-compile-check.sh

set -uo pipefail

PASS=0
FAIL=0
CORE_PASS=0
CORE_FAIL=0

pass() {
    echo "  ✓ PASS: $1"
    PASS=$((PASS + 1))
}

fail() {
    echo "  ✗ FAIL: $1"
    FAIL=$((FAIL + 1))
}

core_pass() {
    pass "$1"
    CORE_PASS=$((CORE_PASS + 1))
}

core_fail() {
    fail "$1"
    CORE_FAIL=$((CORE_FAIL + 1))
}

ensure_target() {
    local target="$1"
    if ! rustup target list --installed | grep -q "$target"; then
        echo "  Installing target $target ..."
        if ! rustup target add "$target" 2>/dev/null; then
            echo "  Could not install target $target — skipping"
            return 1
        fi
    fi
    return 0
}

# ---------- 1. no_std (ARM Cortex-M) — CORE ----------
echo ""
echo "==> [1/5] no_std build (thumbv7em-none-eabihf) — CORE"
TARGET="thumbv7em-none-eabihf"
if ensure_target "$TARGET"; then
    if cargo build --target "$TARGET" -p rustyhdf5-format --no-default-features 2>&1; then
        core_pass "no_std (thumbv7em-none-eabihf)"
    else
        core_fail "no_std (thumbv7em-none-eabihf)"
    fi
else
    core_fail "no_std (thumbv7em-none-eabihf) — target not available"
fi

# ---------- 2. Native host build — CORE ----------
echo ""
echo "==> [2/5] Native host build — CORE"
if cargo build -p rustyhdf5-format 2>&1; then
    core_pass "native host build"
else
    core_fail "native host build"
fi

# ---------- 3. aarch64-unknown-linux-gnu ----------
echo ""
echo "==> [3/5] aarch64-unknown-linux-gnu"
TARGET="aarch64-unknown-linux-gnu"
if ensure_target "$TARGET"; then
    if cargo build --target "$TARGET" -p rustyhdf5-format 2>&1; then
        pass "aarch64-unknown-linux-gnu"
    else
        fail "aarch64-unknown-linux-gnu"
    fi
else
    echo "  SKIP: target $TARGET not available"
fi

# ---------- 4. wasm32-unknown-unknown ----------
echo ""
echo "==> [4/5] wasm32-unknown-unknown"
TARGET="wasm32-unknown-unknown"
if ensure_target "$TARGET"; then
    if cargo build --target "$TARGET" -p rustyhdf5-format --no-default-features 2>&1; then
        pass "wasm32-unknown-unknown"
    else
        fail "wasm32-unknown-unknown"
    fi
else
    echo "  SKIP: target $TARGET not available"
fi

# ---------- 5. x86_64-unknown-linux-gnu ----------
echo ""
echo "==> [5/5] x86_64-unknown-linux-gnu"
TARGET="x86_64-unknown-linux-gnu"
if ensure_target "$TARGET"; then
    if cargo build --target "$TARGET" -p rustyhdf5-format 2>&1; then
        pass "x86_64-unknown-linux-gnu"
    else
        fail "x86_64-unknown-linux-gnu"
    fi
else
    echo "  SKIP: target $TARGET not available"
fi

# ---------- Summary ----------
echo ""
echo "========================================"
echo "  Cross-Compile Summary"
echo "========================================"
echo "  Core targets: $CORE_PASS passed, $CORE_FAIL failed"
echo "  All targets:  $PASS passed, $FAIL failed"
echo "========================================"

if [ "$CORE_FAIL" -gt 0 ]; then
    echo "RESULT: FAIL (core targets failed)"
    exit 1
fi

echo "RESULT: PASS (core targets OK)"
exit 0
