# Locality experiment commands

Run the commands from the repository root. Create the local log directory once:

```bash
mkdir -p logs/locality
```

IMPORTANT: you need to change cu118 for current device setup. 

## 01 — raw SORBES/MUTANG pool

This prepares three runs, one for each configured MUTANG token threshold.

```bash
nohup env PYTHONUNBUFFERED=1 uv run --extra cu118 python \
  scripts/runner/run_optimization.py \
  --config configs/optimization/experiments/locality/01_sorbes_mutang_raw_pool.json \
  > logs/locality/01_sorbes_mutang_raw_pool.log 2>&1 &
echo $! | tee logs/locality/01_sorbes_mutang_raw_pool.pid
```

Follow the log:

```bash
tail -f logs/locality/01_sorbes_mutang_raw_pool.log
```

## 02 — random-position raw baseline

This prepares one run with up to four mutable positions. It serves as control group for testing mutang results 

```bash
nohup env PYTHONUNBUFFERED=1 uv run --extra cu118 python \
  scripts/runner/run_optimization.py \
  --config configs/optimization/experiments/locality/02_random_three_position_raw_pool.json \
  > logs/locality/02_random_three_position_raw_pool.log 2>&1 &
echo $! | tee logs/locality/02_random_three_position_raw_pool.pid
```

Follow the log:

```bash
tail -f logs/locality/02_random_three_position_raw_pool.log
```

## Process status

```bash
ps -fp "$(cat logs/locality/01_sorbes_mutang_raw_pool.pid)"
ps -fp "$(cat logs/locality/02_random_three_position_raw_pool.pid)"
```


