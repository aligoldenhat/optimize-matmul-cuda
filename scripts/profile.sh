#!/usr/bin/env bash
# Profile one kernel with Nsight Compute (ncu) and save a report for ncu-ui.
#
#   scripts/profile.sh [kernel-name-regex] [size] [report-name]
#   scripts/profile.sh sgemm_shared 1024 k3        # -> build/k3.ncu-rep
#
# ncu needs access to the GPU's performance counters, which on this machine
# requires root, so the script runs it with sudo (it will ask for your password).
# sudo has its own PATH and wouldn't find ncu, so we pass the full path.
# The report is chown'ed back to you, so ncu-ui can open it without sudo.
set -euo pipefail
cd "$(dirname "$0")/.."

KERNEL=${1:-sgemm_shared}  # regex, matched against the kernel's function name
SIZE=${2:-1024}            # square matrix size, a multiple of 32
NAME=${3:-profile}
BIN=build/release/sgemm
REPORT=build/$NAME.ncu-rep

if [[ ! -x $BIN ]]; then
    echo "Build first: cmake --preset release && cmake --build --preset release" >&2
    exit 1
fi

NCU=$(command -v ncu)

# --set full         : collect every section (stall reasons, memory, occupancy, ...)
# --import-source on : store your source code inside the report
# --launch-count 1   : profile only the first launch that matches the name
sudo "$NCU" --set full --import-source on \
    --kernel-name "regex:${KERNEL}.*" --launch-count 1 \
    --force-overwrite -o "build/$NAME" "$BIN" "$SIZE"

sudo chown "$USER" "$REPORT"
echo "Report written: $REPORT"
echo "Open it with:   ncu-ui $REPORT"
