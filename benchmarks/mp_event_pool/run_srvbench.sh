#!/bin/bash
# Run srvbench.py against one LMCache MP server configuration.
#
# usage: run_srvbench.sh OUT [pool] [extra server args...]
#   OUT   output directory (server.log, sweep.json, burst.json)
#   pool  "1" sets LMCACHE_MP_POOLED_EVENTS=1 for the clients (needs the pooled-events
#         patch); anything else runs the stock event path
#   extra server args, e.g. --max-gpu-workers 4
#
# The LMCache under test is whatever `import lmcache` resolves to; point PYTHONPATH at a
# different build to compare versions. Uses 4 GPUs (HIP_VISIBLE_DEVICES selects them).
# env: SB_PORT (15566), SB_PROM (Prometheus port, 19788), SB_L1 (L1 size in GB, 1200),
#      SB_LAZY (1 = --l1-use-lazy; 0 allocates L1 up front, avoids lazy-growth stalls),
#      SB_T_SWEEP, SB_BURSTS, SB_BURST_R
set -uo pipefail
OUT=$1; POOL=${2:-0}; shift; [ $# -gt 0 ] && shift
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${SB_PORT:-15566}; PROM=${SB_PROM:-19788}
port_busy() { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null; }
for _ in $(seq 120); do port_busy $PORT || port_busy $PROM || break; sleep 1; done
if port_busy $PORT || port_busy $PROM; then echo "port $PORT or $PROM busy"; exit 1; fi
rm -rf "$OUT"; mkdir -p "$OUT"
export LMCACHE_MP_POOLED_EVENTS=$([ "$POOL" = 1 ] && echo 1 || echo 0)
python -c 'import lmcache; print(lmcache.__version__, lmcache.__file__)' 2>/dev/null | tail -1 > "$OUT/lmcache_version.txt"
echo "server args: $*  pooled=$LMCACHE_MP_POOLED_EVENTS  lmcache: $(cat "$OUT/lmcache_version.txt")" | tee "$OUT/config.txt"

python -m lmcache.v1.multiprocess.server --host 127.0.0.1 --port "$PORT" --chunk-size 256 \
  --null-block-id -1 --separate-object-groups --supported-transfer-mode lmcache_driven \
  --l1-size-gb "${SB_L1:-1200}" $([ "${SB_LAZY:-1}" = 1 ] && echo --l1-use-lazy) --l1-read-ttl-seconds 900 --eviction-policy LRU \
  --eviction-trigger-watermark 0.98 --eviction-ratio 0.01 --event-bus-queue-size 4000000 --prometheus-port "$PROM" "$@" \
  > "$OUT/server.log" 2>&1 &
SP=$!
until grep -qE "cache server is running|Traceback" "$OUT/server.log" 2>/dev/null; do
  sleep 2; kill -0 $SP 2>/dev/null || break
done
if grep -q Traceback "$OUT/server.log"; then echo "server failed"; tail -n 20 "$OUT/server.log"; kill $SP; exit 1; fi

for phase in sweep burst; do
  timeout 3000 python "$HERE/srvbench.py" "$OUT" "$PORT" $phase > "$OUT/$phase.log" 2>&1
  echo "$phase exit $?" | tee -a "$OUT/$phase.log"
done
kill -INT $SP; for _ in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
for _ in $(seq 60); do port_busy $PORT || port_busy $PROM || break; sleep 1; done
python "$HERE/analyze.py" "$OUT"
