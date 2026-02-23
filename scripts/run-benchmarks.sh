#!/usr/bin/env bash
# Run Criterion benchmarks for rustyhdf5-format and generate a markdown report.
#
# Usage:
#   ./scripts/run-benchmarks.sh
#
# Output:
#   BENCHMARKS.md in the repository root

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT="$REPO_ROOT/BENCHMARKS.md"
BENCH_OUTPUT=$(mktemp)

echo "==> Running benchmarks for rustyhdf5-format ..."
cargo bench -p rustyhdf5-format 2>&1 | tee "$BENCH_OUTPUT"
BENCH_EXIT=${PIPESTATUS[0]}

if [ "$BENCH_EXIT" -ne 0 ]; then
    echo "ERROR: cargo bench failed with exit code $BENCH_EXIT"
    rm -f "$BENCH_OUTPUT"
    exit 1
fi

# Parse criterion output lines like:
#   bench_name          time:   [1.234 ms 1.256 ms 1.278 ms]
# We extract the middle (point estimate) value.
declare -a NAMES=()
declare -a TIMES=()

while IFS= read -r line; do
    if [[ "$line" =~ ^([a-zA-Z0-9_/]+)[[:space:]]+time:[[:space:]]+\[.*[[:space:]]+([-0-9.]+[[:space:]]+(ns|µs|us|μs|ms|s))[[:space:]]+.*\] ]]; then
        NAMES+=("${BASH_REMATCH[1]}")
        TIMES+=("${BASH_REMATCH[2]}")
    fi
done < "$BENCH_OUTPUT"

# Generate report
{
    echo "# rustyhdf5-format Benchmark Results"
    echo ""
    echo "Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo ""
    echo "## System Info"
    echo ""
    echo "- **OS**: $(uname -srm)"
    echo "- **Rust**: $(rustc --version)"
    echo "- **CPU**: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: *//' || echo 'unknown')"
    echo ""
    echo "## Results"
    echo ""
    echo "| Benchmark | Time (point estimate) |"
    echo "|-----------|----------------------|"

    for i in "${!NAMES[@]}"; do
        echo "| ${NAMES[$i]} | ${TIMES[$i]} |"
    done

    if [ "${#NAMES[@]}" -eq 0 ]; then
        echo "| (no results parsed — see raw output below) | — |"
    fi

    echo ""
    echo "## Notes"
    echo ""
    echo "- All benchmarks use Criterion.rs with default settings."
    echo "- 1M dataset = 1,000,000 f64 values (~7.6 MB)."
    echo "- Chunked benchmarks use 10K-element chunks."
    echo "- Run with: \`./scripts/run-benchmarks.sh\`"
} > "$REPORT"

rm -f "$BENCH_OUTPUT"

echo ""
echo "==> Benchmark report written to $REPORT"
echo "==> $(( ${#NAMES[@]} )) benchmarks captured."
