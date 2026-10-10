#!/usr/bin/env bash
cd "$(dirname "$0")/../results/gate0/logs"
echo "files with Traceback: $(grep -l Traceback *.log | wc -l)"
echo "files with Killed:    $(grep -l Killed *.log | wc -l)"
echo "files with OOM text:  $(grep -l OOM *.log | wc -l)"
grep -h -o 'memory allocation failed with OOM.\{0,80\}' *.log | head -2
echo "training runs ending with DONE: $(grep -l '^DONE:' train_*.log | wc -l) of $(ls train_*.log | wc -l)"
echo "steps per finished run (expect 376):"
grep -h '^DONE:' train_*.log | sed 's/.*steps=\([0-9]*\).*/\1/' | sort | uniq -c
