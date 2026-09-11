#!/bin/bash
# Per-case ETACE trace test with an RSS watchdog: a case that blows up is
# killed at 25 GB instead of taking the whole machine down (a struct-array KA
# kernel reached 60 GB and OOM-killed the box on the first attempt).
cd /home/eng/essswb/reactant_etace
LIMIT_KB=26214400          # 25 GiB
run_case () {
  local c=$1 b=$2
  echo "--- case=$c backend=$b ---"
  julia +1.11 --project=. etace_one.jl $c $b > /tmp/et.$c.$b 2>&1 &
  local pid=$! peak=0
  while kill -0 $pid 2>/dev/null; do
    local rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
    [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
    if [ -n "$rss" ] && [ "$rss" -gt "$LIMIT_KB" ]; then
      kill -9 $pid 2>/dev/null
      echo "WATCHDOG: killed at RSS ${rss} kB (> 25 GiB)"
      break
    fi
    sleep 3
  done
  wait $pid 2>/dev/null; local ec=$?
  grep -E "^RESULT|^   AT |^## " /tmp/et.$c.$b
  echo "exitcode=$ec  peakRSS=$((peak/1024))MB"
  grep -q "^RESULT" /tmp/et.$c.$b || { echo "NO RESULT -- process died. tail:"; tail -4 /tmp/et.$c.$b; }
}
for b in "$@"; do
  for c in yembed rembed_nosel abasis aabasis rembed site_basis site_basis_R sitee; do
    run_case $c $b
  done
done
echo ALLDONE
