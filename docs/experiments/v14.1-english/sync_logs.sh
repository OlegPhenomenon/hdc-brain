#!/bin/bash
# Sync training logs from vast.ai server to local machine
# Usage: ./sync_logs.sh

SERVER="root@87.205.21.33"
PORT=42047
LOCAL_DIR="$(dirname "$0")"

echo "Syncing from $SERVER:$PORT..."

scp -P $PORT $SERVER:/workspace/experiment.jsonl "$LOCAL_DIR/experiment.jsonl" 2>/dev/null
scp -P $PORT $SERVER:/workspace/train.log "$LOCAL_DIR/train.log" 2>/dev/null
scp -P $PORT $SERVER:/workspace/stdout.log "$LOCAL_DIR/stdout.log" 2>/dev/null
scp -P $PORT $SERVER:/workspace/train.py "$LOCAL_DIR/train.py" 2>/dev/null

echo ""
echo "=== Status ==="
if [ -f "$LOCAL_DIR/experiment.jsonl" ]; then
    ENTRIES=$(wc -l < "$LOCAL_DIR/experiment.jsonl")
    LAST_EVENT=$(tail -1 "$LOCAL_DIR/experiment.jsonl" | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'{d[\"event\"]} iter={d.get(\"iter\",\"?\")}  {d.get(\"timestamp\",\"\")[:19]}')" 2>/dev/null)
    echo "JSONL entries: $ENTRIES"
    echo "Last event:    $LAST_EVENT"
fi

if [ -f "$LOCAL_DIR/train.log" ]; then
    echo "Train log:     $(wc -l < "$LOCAL_DIR/train.log") lines"
    echo "Last line:     $(tail -1 "$LOCAL_DIR/train.log")"
fi

echo ""
echo "Files saved to: $LOCAL_DIR/"
