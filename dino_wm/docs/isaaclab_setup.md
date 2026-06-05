# IsaacLab setup

Opt-in via `ISAACLAB_AVAILABLE=1`. Run inside the Isaac Sim docker.

## Required env vars
- `ISAACLAB_AVAILABLE=1` — registers the `isaaclab_grid` gym id.
- `DATASET_DIR` — host path for datasets; bind-mounted into the container at the same path.

## Container launch (sketch)
```
docker run --gpus all \
  -v /newdata2/dylantw/Legislative-Harness/dino_wm:/workspace/dino_wm \
  -v $DATASET_DIR:$DATASET_DIR \
  -e DATASET_DIR -e ISAACLAB_AVAILABLE=1 \
  <isaac_sim_image>
```

## Inside the container
- Python 3.11.
- `./IsaacLab/isaaclab.sh --install` once.
- Install dino_wm deps for 3.11 (the repo's `environment.yaml` is 3.9 host-only).

## num_envs
GPU-batched, one process, one GPU. Start at 16, halve on OOM.
