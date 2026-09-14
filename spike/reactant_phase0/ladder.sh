#!/bin/bash
# Drive ka_one.jl over every case x both Reactant backends, one process per
# case so a hard crash is attributable and cannot swallow the results after it.
#
# WARNING: case B1b (a host array of structs passed into a KA kernel) makes
# Reactant allocate without bound -- it reached 60 GB and OOM-killed a 62 GB
# machine, taking sshd with it. The RSS watchdog below kills any case at 25 GiB.
# Do not remove it when running on a shared box.
cd "$(dirname "$0")"
LIMIT_KB=26214400          # 25 GiB
run_case () {
  local c=$1 b=$2
  echo "--- case=$c backend=$b ---"
  julia +1.11 --project=. ka_one.jl "$c" "$b" > /tmp/ka.$c.$b 2>&1 &
  local pid=$! peak=0 rss=""
  while kill -0 $pid 2>/dev/null; do
    rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
    if [ -n "$rss" ]; then
      [ "$rss" -gt "$peak" ] && peak=$rss
      if [ "$rss" -gt "$LIMIT_KB" ]; then
        kill -9 $pid 2>/dev/null
        echo "WATCHDOG: killed at RSS ${rss} kB (> 25 GiB)"
        break
      fi
    fi
    sleep 3
  done
  wait $pid 2>/dev/null; local ec=$?
  grep -E "^RESULT|^## " /tmp/ka.$c.$b
  echo "exitcode=$ec  peakRSS=$((peak/1024))MB"
  grep -q "^RESULT" /tmp/ka.$c.$b || { echo "NO RESULT -- died. tail:"; tail -4 /tmp/ka.$c.$b; }
}
for b in "${@:-cpu gpu}"; do
  for c in 0b 1 2 3 4 B1 B1b B2 C D; do
    run_case "$c" "$b"
  done
done
echo ALLDONE
