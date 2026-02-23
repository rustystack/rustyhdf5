#!/usr/bin/env bash
# =============================================================================
# audit_cargo_tomls.sh
# Run from workspace root BEFORE publish_prep.sh
# Prints exactly what needs to change in each Cargo.toml, with ready-to-paste fixes.
# Does NOT modify files — shows you the diffs to apply yourself.
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()    { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()     { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()  { echo -e "${RED}[FIX]${NC}  $*"; }

# ─── Config — adjust for your repos ──────────────────────────────────────────
RUSTYHDF5_REPO="https://github.com/rustystack/rustyhdf5"
EDGEHDF5_REPO="https://github.com/rustystack/edgehdf5"
LICENSE="MIT"
# Detect which repo we're in
if grep -rq 'rustyhdf5' Cargo.toml 2>/dev/null || ls crates/ 2>/dev/null | grep -q 'rustyhdf5'; then
  REPO_URL="$RUSTYHDF5_REPO"
  echo -e "${BOLD}Detected: rustyhdf5 workspace${NC}"
elif grep -rq 'edgehdf5' Cargo.toml 2>/dev/null || ls crates/ 2>/dev/null | grep -q 'edgehdf5'; then
  REPO_URL="$EDGEHDF5_REPO"
  echo -e "${BOLD}Detected: edgehdf5 workspace${NC}"
else
  REPO_URL="https://github.com/rustystack/UNKNOWN"
  warn "Could not detect repo — set REPO_URL manually in this script"
fi
# ─────────────────────────────────────────────────────────────────────────────

echo ""

# Descriptions per known crate name — add yours here
declare -A DESCRIPTIONS=(
  # rustyhdf5
  ["rustyhdf5-types"]="HDF5 type system definitions for the rustyhdf5 pure-Rust HDF5 library"
  ["rustyhdf5-format"]="Pure-Rust HDF5 binary format parsing and writing (no_std compatible)"
  ["rustyhdf5-io"]="Memory-mapped and buffered I/O backends for rustyhdf5"
  ["rustyhdf5-filters"]="Compression filter pipeline (deflate, shuffle, fletcher32) for rustyhdf5"
  ["rustyhdf5-accel"]="SIMD acceleration primitives (NEON, AVX2, AVX-512) for rustyhdf5"
  ["rustyhdf5-gpu"]="GPU compute via wgpu for rustyhdf5"
  ["rustyhdf5-derive"]="Procedural macros for deriving HDF5 traits in rustyhdf5"
  ["rustyhdf5"]="Pure-Rust HDF5 reader and writer — zero C dependencies, h5py compatible"
  ["rustyhdf5-netcdf4"]="NetCDF-4 compatibility layer built on rustyhdf5"
  ["rustyhdf5-ann"]="HNSW approximate nearest-neighbor index stored in HDF5 via rustyhdf5"
  ["rustyhdf5-py"]="Python bindings for rustyhdf5 via PyO3"
  # edgehdf5
  ["edgehdf5-memory"]="HDF5-backed persistent memory store for on-device AI agents"
  ["edgehdf5-migrate"]="CLI to migrate SQLite agent memory databases to HDF5 format"
)

# Keywords per crate
declare -A KEYWORDS=(
  ["rustyhdf5-types"]="hdf5 scientific storage types no-std"
  ["rustyhdf5-format"]="hdf5 binary format parsing no-std scientific"
  ["rustyhdf5-io"]="hdf5 io mmap buffered scientific"
  ["rustyhdf5-filters"]="hdf5 compression deflate gzip scientific"
  ["rustyhdf5-accel"]="simd avx2 neon acceleration hdf5"
  ["rustyhdf5-gpu"]="gpu wgpu hdf5 acceleration compute"
  ["rustyhdf5-derive"]="hdf5 derive macros scientific"
  ["rustyhdf5"]="hdf5 scientific data storage no-std"
  ["rustyhdf5-netcdf4"]="netcdf4 hdf5 scientific climate"
  ["rustyhdf5-ann"]="hnsw ann vector-search hdf5 embedding"
  ["rustyhdf5-py"]="hdf5 python pyo3 scientific"
  ["edgehdf5-memory"]="agent memory vector-search hdf5 embedding"
  ["edgehdf5-migrate"]="sqlite hdf5 migration agent memory"
)

# Categories (must be from https://crates.io/categories)
declare -A CATEGORIES=(
  ["rustyhdf5-types"]="encoding science"
  ["rustyhdf5-format"]="encoding science no-std"
  ["rustyhdf5-io"]="filesystem encoding"
  ["rustyhdf5-filters"]="compression encoding"
  ["rustyhdf5-accel"]="algorithms science"
  ["rustyhdf5-gpu"]="rendering science"
  ["rustyhdf5-derive"]="development-tools"
  ["rustyhdf5"]="encoding science filesystem"
  ["rustyhdf5-netcdf4"]="encoding science"
  ["rustyhdf5-ann"]="algorithms science"
  ["rustyhdf5-py"]="api-bindings science"
  ["edgehdf5-memory"]="database science algorithms"
  ["edgehdf5-migrate"]="command-line-utilities database"
)

# ─── Scan all crates ──────────────────────────────────────────────────────────

TOTAL_ISSUES=0

while IFS= read -r -d '' toml; do
  dir=$(dirname "$toml")
  [[ "$dir" == "." ]] && continue  # skip root virtual manifest

  if ! grep -q '^\[package\]' "$toml" 2>/dev/null; then continue; fi

  cname=$(grep '^name' "$toml" | head -1 | sed 's/name\s*=\s*"\(.*\)"/\1/' | tr -d ' ')
  version=$(grep '^version' "$toml" | head -1 | sed 's/version\s*=\s*"\(.*\)"/\1/' | tr -d ' ')
  crate_issues=0

  echo -e "${BOLD}━━━ $cname @ $version ($dir) ━━━${NC}"

  # ── description ──────────────────────────────────────────────────────────
  if ! grep -q '^description' "$toml"; then
    desc="${DESCRIPTIONS[$cname]:-"A crate in the rustyhdf5 ecosystem"}"
    error "Missing description. Add to [package] in $toml:"
    echo -e "  ${YELLOW}description = \"$desc\"${NC}"
    ((crate_issues++))
  else
    ok "description present"
  fi

  # ── license ───────────────────────────────────────────────────────────────
  if ! grep -q '^license' "$toml"; then
    error "Missing license. Add:"
    echo -e "  ${YELLOW}license = \"$LICENSE\"${NC}"
    ((crate_issues++))
  else
    ok "license present"
  fi

  # ── repository ────────────────────────────────────────────────────────────
  if ! grep -q '^repository' "$toml"; then
    error "Missing repository. Add:"
    echo -e "  ${YELLOW}repository = \"$REPO_URL\"${NC}"
    ((crate_issues++))
  else
    ok "repository present"
  fi

  # ── keywords (recommended, max 5) ────────────────────────────────────────
  if ! grep -q '^keywords' "$toml"; then
    kw="${KEYWORDS[$cname]:-"hdf5 rust"}"
    # Format as TOML array
    kw_array=$(echo "$kw" | tr ' ' '\n' | head -5 | awk '{printf "\"%s\", ", $0}' | sed 's/, $//')
    warn "Missing keywords (recommended). Add:"
    echo -e "  ${YELLOW}keywords = [$kw_array]${NC}"
  else
    ok "keywords present"
  fi

  # ── categories (recommended) ──────────────────────────────────────────────
  if ! grep -q '^categories' "$toml"; then
    cats="${CATEGORIES[$cname]:-"encoding"}"
    cat_array=$(echo "$cats" | tr ' ' '\n' | head -5 | awk '{printf "\"%s\", ", $0}' | sed 's/, $//')
    warn "Missing categories (recommended). Add:"
    echo -e "  ${YELLOW}categories = [$cat_array]${NC}"
  else
    ok "categories present"
  fi

  # ── edition ───────────────────────────────────────────────────────────────
  if ! grep -q '^edition' "$toml"; then
    warn "Missing edition. Add:"
    echo -e "  ${YELLOW}edition = \"2021\"${NC}"
  else
    ok "edition present"
  fi

  # ── publish = false (blocker) ─────────────────────────────────────────────
  if grep -q 'publish\s*=\s*false' "$toml"; then
    error "publish = false is set — REMOVE this line or publishing will be blocked"
    ((crate_issues++))
  fi

  # ── path deps without version ─────────────────────────────────────────────
  echo ""
  log "Checking path dependencies..."
  in_deps_section=false
  while IFS= read -r line; do
    # Track if we're in a [dependencies] section
    if echo "$line" | grep -qP '^\[(dependencies|dev-dependencies|build-dependencies)\]'; then
      in_deps_section=true
    elif echo "$line" | grep -qP '^\[' && ! echo "$line" | grep -qP '^\[dependencies'; then
      in_deps_section=false
    fi

    if echo "$line" | grep -q 'path\s*='; then
      if ! echo "$line" | grep -q 'version\s*='; then
        dep_name=$(echo "$line" | grep -oP '^\s*\K[\w-]+(?=\s*=)')
        dep_path=$(echo "$line" | grep -oP 'path\s*=\s*"\K[^"]+')
        error "Path dep '$dep_name' has no version — crates.io will REJECT this."
        echo -e "  Change:"
        echo -e "  ${RED}  $line${NC}"
        echo -e "  To:"
        echo -e "  ${YELLOW}  $dep_name = { path = \"$dep_path\", version = \"$version\" }${NC}"
        echo -e "  (use the actual version of $dep_name)"
        ((crate_issues++))
      else
        dep_name=$(echo "$line" | grep -oP '^\s*\K[\w-]+(?=\s*=)')
        ok "Path dep '$dep_name' has version — OK"
      fi
    fi
  done < "$toml"

  # ── git deps (not allowed on crates.io) ──────────────────────────────────
  if grep -q 'git\s*=' "$toml"; then
    error "Git dependencies found — crates.io DOES NOT allow git deps:"
    grep -n 'git\s*=' "$toml" | while read -r gitline; do
      echo -e "  ${RED}$gitline${NC}"
    done
    echo "  You must either:"
    echo "    a) Publish the git dep to crates.io first, then reference by version"
    echo "    b) Vendor it into the repo"
    ((crate_issues++))
  fi

  # ── check README exists ───────────────────────────────────────────────────
  if grep -q '^readme' "$toml"; then
    readme_val=$(grep '^readme' "$toml" | sed 's/readme\s*=\s*"\(.*\)"/\1/' | tr -d ' ')
    readme_path="$dir/$readme_val"
    if [[ -f "$readme_path" ]]; then
      ok "README found: $readme_path"
    else
      warn "README listed as '$readme_val' but not found at $readme_path"
    fi
  else
    # Check if README.md exists even without being declared
    if [[ -f "$dir/README.md" ]]; then
      warn "README.md exists but not declared. Add:"
      echo -e "  ${YELLOW}readme = \"README.md\"${NC}"
    fi
  fi

  # ── Summary for this crate ────────────────────────────────────────────────
  echo ""
  if [[ $crate_issues -eq 0 ]]; then
    ok "$cname — NO blocking issues found ✓"
  else
    error "$cname — $crate_issues blocking issue(s) to fix before publishing"
  fi
  TOTAL_ISSUES=$((TOTAL_ISSUES + crate_issues))
  echo ""

done < <(find . -name "Cargo.toml" -not -path "*/target/*" -print0 | sort -z)

# ─── Final summary ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━ SUMMARY ━━━${NC}"
if [[ $TOTAL_ISSUES -eq 0 ]]; then
  echo -e "${GREEN}All crates are ready for publishing! Run publish_prep.sh next.${NC}"
else
  echo -e "${RED}$TOTAL_ISSUES total blocking issue(s) found across all crates.${NC}"
  echo ""
  echo "  Fix all [FIX] items above, then run this script again to confirm."
  echo "  [WARN] items are recommended but won't block publishing."
  echo ""
  echo "  Quick guide to editing Cargo.toml:"
  echo "    vim crates/<name>/Cargo.toml"
  echo "    # or"
  echo "    code crates/<name>/Cargo.toml"
fi
echo ""
