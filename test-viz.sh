#!/bin/bash
# Engram viz test script — stores memories, waits, then cleans up
# Run this while `engram viz` is open in another terminal
set -e

ENGRAM="${ENGRAM_BIN:-$HOME/.engram/bin/engram}"
SOURCE="viz-test-$(date +%s)"

cleanup() {
    echo ""
    echo "Cleaning up..."
    $ENGRAM forget --source "%${SOURCE}%" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

echo "=== Engram Viz Test ==="
echo "Make sure 'engram viz' is running in another terminal."
echo ""

echo "1) Storing memories..."
$ENGRAM store "Viz test: Alice works on the database cluster" --source "$SOURCE"
sleep 2
$ENGRAM store "Viz test: Bob manages the API gateway" --source "$SOURCE"
sleep 2
$ENGRAM store "Viz test: Charlie deployed the monitoring stack" --source "$SOURCE"

echo ""
echo "2) Waiting 5s for you to observe..."
sleep 5

echo ""
echo "3) Deleting all test memories..."
$ENGRAM forget --source "%${SOURCE}%"

echo ""
echo "4) Waiting 5s for you to observe deletion..."
sleep 5

echo ""
echo "Test complete."