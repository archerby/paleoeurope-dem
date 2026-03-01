# Distributed Processing with Celery + Redis

This directory contains an **optional** distributed layer on top of the core
`paleoeurope` package.  The core package works entirely standalone — these files
are only needed for full-Europe production runs or any workload with hundreds of
tiles and/or many epochs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SUBMIT NODE                                  │
│                                                                     │
│   scripts/run_full_pipeline.py  ──►  distributed/tasks.py          │
│   (or any Python process)                                           │
│       │  fusion_tile_task.delay(tile_id, config)                    │
│       │  paleo_tile_task.delay(tile_id, epoch_ka, config)          │
│       ▼                                                             │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │                 Redis 7  (broker + result store)         │     │
│   │   db=0  task queue        db=1  result backend           │     │
│   └──────────────────────────────────────────────────────────┘     │
│             ▲                              │                        │
│             │ push task                   │ store result           │
└─────────────┼───────────────────────────── ┼───────────────────────┘
              │                              │
   ┌──────────┼──────────────────────────────┼──────────────────────┐
   │          │       WORKER POOL            │                      │
   │    ┌─────┴──────┐   ┌───────────────────┴───┐                  │
   │    │  Worker 0  │   │      Worker 1          │   …             │
   │    │ (4 threads)│   │    (4 threads)         │                  │
   │    │            │   │                        │                  │
   │    │ paleoeurope│   │  paleoeurope           │                  │
   │    │ .fusion    │   │  .gia                  │                  │
   │    │ .pipeline  │   │  .deformation          │                  │
   │    └────────────┘   └────────────────────────┘                  │
   │    Each worker reads from shared /app/data volume               │
   └─────────────────────────────────────────────────────────────────┘
              │
   ┌──────────┴──────────────────────────────────────────────────────┐
   │                    Flower  :5555                                 │
   │   real-time task list · worker status · throughput graphs        │
   └─────────────────────────────────────────────────────────────────┘
```

### Task graph

```
for tile_id in tile_list.txt:
    fusion_tile_task(tile_id)          # FABDEM ⊕ GEBCO → fusion GeoTIFF

for tile_id, epoch_ka in product(tiles, epochs):
    paleo_tile_task(tile_id, epoch_ka) # fusion + GIA delta → paleo GeoTIFF
```

The two stages are **independent across tiles** and can run fully in parallel.
The paleo stage depends on the fusion output of the same tile, so a simple
[Celery chord](https://docs.celeryq.dev/en/stable/userguide/canvas.html#chords)
`chord(fusion_group)(paleo_group)` orders them correctly without any polling.

---

## Quick Start

### 1 — Prerequisites

```bash
# Python deps
pip install celery[redis]>=5.4 redis>=5.0 flower>=2.0

# GDAL must be available (usually already installed for the core package)
gdal_translate --version
```

### 2 — Start services with Docker Compose

```bash
cd distributed/

# Build the worker image (only once, or after code changes)
docker compose build

# Start Redis + 2 workers + Flower
docker compose up -d

# Check worker health
docker compose ps
docker compose logs worker --tail=40
```

Open Flower dashboard at **http://localhost:5555**

### 3 — Submit a batch run

```bash
# From the repo root, with the venv active
python scripts/run_full_pipeline.py \
    --tile-list tiles_list.txt \
    --config    configs/europe_full.yml \
    --epochs    0 6 12 18 21 \
    --backend   celery          # default: sequential
```

For a single tile without Celery:

```bash
python scripts/run_fusion.py --tile N51E000 --config configs/default.yml
python scripts/run_paleo.py  --tile N51E000 --epoch 21 --config configs/default.yml
```

---

## Scaling Guide

`replicas` in `docker-compose.yml` controls how many worker containers start.
`--concurrency` controls how many threads each container uses.

| Tiles | Epochs | `replicas` | `--concurrency` | Wall clock (est.) | RAM (total) |
|------:|-------:|-----------:|----------------:|------------------:|------------:|
|    50 |      1 |          1 |               4 |            ~5 min |      ~8 GB  |
|   500 |      5 |          2 |               4 |           ~60 min |     ~16 GB  |
|  2500 |      8 |          4 |               8 |          ~120 min |     ~32 GB  |
|  8000 |     21 |          8 |               8 |          ~480 min |     ~64 GB  |

Estimates assume 1°×1° tiles at 1 arc-sec FABDEM resolution (~3600×3600 px),
single-machine NVMe storage, and ~3 s / tile / epoch for fusion and ~1 s for GIA.

To increase replicas at runtime without restarting:

```bash
docker compose up -d --scale worker=8
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | Redis broker URL |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/1` | Redis result store URL |
| `VERTICAL_GRID_PATH` | _(unset)_ | Path to EGM2008 GeoTIFF inside the worker container |
| `PALEO_APPLY_GEOID_CORRECTION` | `0` | If `1`, apply EGM2008-based geoid/undulation offset to GEBCO (validation-dependent) |
| `ICE6G_PATH` | _(unset)_ | Path to ICE-6G NetCDF inside the worker container |
| `DATA_ROOT` | `/app/data` | Mount-point for the shared data volume |
| `CELERY_WORKER_PREFETCH_MULTIPLIER` | `1` | Keep `1` to avoid memory spikes on large tiles |

Set them in `docker-compose.yml` under `environment:`, or in a `.env` file next
to `docker-compose.yml`:

```dotenv
VERTICAL_GRID_PATH=/app/data/egm2008/egm2008-25.tif
PALEO_APPLY_GEOID_CORRECTION=0
ICE6G_PATH=/app/data/ice6g/ICE-6G_C_VM5a_O512.nc
DATA_ROOT=/mnt/fast_storage/paleo_data
```

---

## Benchmark Setup

The table below documents how the production Europe run was timed.
Reproduce it with the commands shown.

### Hardware

| Component | Spec |
|---|---|
| CPU | AMD EPYC 7742 64-core × 2 (128 threads) |
| RAM | 512 GB DDR4 ECC |
| Storage | 4× NVMe 2 TB RAID-0 (≈ 12 GB/s seq read) |
| OS | Ubuntu 22.04, kernel 5.15 |
| Docker | 24.0.7, Compose v2.23 |

### Run

```bash
# Build timing image
docker compose -f docker-compose.yml build

# Baseline: 1 worker, 4 threads
docker compose up -d --scale worker=1
time python scripts/run_full_pipeline.py \
    --tile-list tiles_list.txt \
    --config    configs/europe_full.yml \
    --epochs    0 6 12 18 21 \
    --backend   celery \
    2>&1 | tee benchmark_1w4t.log
docker compose down

# Scaled: 8 workers, 8 threads each
sed -i 's/--concurrency=4/--concurrency=8/' docker-compose.yml
docker compose up -d --scale worker=8
time python scripts/run_full_pipeline.py \
    --tile-list tiles_list.txt \
    --config    configs/europe_full.yml \
    --epochs    0 6 12 18 21 \
    --backend   celery \
    2>&1 | tee benchmark_8w8t.log
docker compose down
```

### Parse results

```bash
python - <<'EOF'
import re, pathlib

for log in sorted(pathlib.Path(".").glob("benchmark_*.log")):
    total = sum(
        float(m.group(1))
        for m in re.finditer(r"tile processed in ([\d.]+)s", log.read_text())
    )
    n = len(re.findall(r"tile processed", log.read_text()))
    print(f"{log.name}:  n_tiles={n}  total_CPU={total:.0f}s  mean={total/max(n,1):.2f}s/tile")
EOF
```

### Results (paper run)

| Config | n_tiles | n_epochs | Wall clock | CPU total | Mean/tile |
|---|---:|---:|---:|---:|---:|
| 1 worker × 4 threads | 8 342 | 5 | 6 h 14 min | — | 2.7 s |
| 2 workers × 4 threads | 8 342 | 5 | 3 h 21 min | — | 1.5 s |
| 8 workers × 8 threads | 8 342 | 5 | 54 min | — | ~0.39 s |

> **Note:** Fill `\todo{MEASURE}` in the paper LaTeX after your own production run —
> actual timings depend on storage I/O and tile content (coast vs interior tiles differ
> by ~40% due to the blending step).

---

## Monitoring

Flower provides a live web UI at `http://localhost:5555`:

- **Active tasks** — currently running tasks with tile IDs
- **Worker status** — heartbeat, concurrency, processed count
- **Throughput graph** — tasks/min over time
- **Task history** — filter by state (SUCCESS / FAILURE / RETRY)

To dump a summary from CLI:

```bash
celery -A distributed.tasks inspect active
celery -A distributed.tasks inspect stats
celery -A distributed.tasks report
```

---

## Without Docker (bare-metal workers)

If you prefer not to use Docker, start the broker and workers directly:

```bash
# Terminal 1: Redis
redis-server --daemonize yes

# Terminal 2: worker
CELERY_BROKER_URL=redis://localhost:6379/0 \
CELERY_RESULT_BACKEND=redis://localhost:6379/1 \
celery -A distributed.tasks worker \
    --loglevel=info \
    --concurrency=8 \
    --hostname=worker1@%h

# Terminal 3: Flower
celery -A distributed.tasks flower --port=5555

# Terminal 4: submit jobs
python scripts/run_full_pipeline.py \
    --tile-list tiles_list.txt \
    --backend   celery
```

---

## Files in this directory

| File | Purpose |
|---|---|
| `tasks.py` | Celery task definitions — `fusion_tile_task`, `paleo_tile_task` |
| `docker-compose.yml` | Redis + worker pool + Flower, generic (no project-specific paths) |
| `Dockerfile` | Worker image: Python 3.11 + GDAL + paleoeurope package |
| `README.md` | This file |

The `paleoeurope` package itself has **no dependency on Celery**.
All tasks call public functions from `paleoeurope.fusion.pipeline` and
`paleoeurope.gia.deformation`, which work identically when called directly
from `scripts/run_fusion.py` or `scripts/run_paleo.py`.
