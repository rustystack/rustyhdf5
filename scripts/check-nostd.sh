#!/usr/bin/env bash
# CI check: verify rustyhdf5-format compiles under no_std (thumbv7em-none-eabihf).
#
# Usage:
#   ./scripts/check-nostd.sh
#
# Prerequisites:
#   rustup target add thumbv7em-none-eabihf

set -euo pipefail

TARGET="thumbv7em-none-eabihf"

echo "==> Checking no_std build for rustyhdf5-format (target: $TARGET)"

# Ensure the target is installed
if ! rustup target list --installed | grep -q "$TARGET"; then
    echo "Installing target $TARGET ..."
    rustup target add "$TARGET"
fi

# Build with no default features (no std, no flate2, no sha2)
cargo build --target "$TARGET" -p rustyhdf5-format --no-default-features

echo "==> no_std build succeeded"

# Also verify the default-features (std) build still works
echo "==> Checking default-features build"
cargo build -p rustyhdf5-format
echo "==> default-features build succeeded"

echo "==> All no_std checks passed"
