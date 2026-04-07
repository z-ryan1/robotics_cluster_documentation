# Contact-Rich Gear Assembly Training on Stampede3 (8-GPU)

Run the full Isaac Lab RL training for the contact-rich gear assembly task using all 8 RTX PRO 6000 Blackwell GPUs on a single `amd-rtx` node.

## Prerequisites

1. **Isaac Lab container pulled** — follow the [Isaac Lab NGC Container guide](../tacc_stampede_isaaclab_container/) to pull `nvcr.io/nvidia/isaac-lab:2.3.2` to `$SCRATCH/containers/isaaclab/isaac-lab_2.3.2.sif`.

2. **`amd-rtx` allocation** — confirm you have an active allocation:

   ```bash
   /usr/local/etc/taccinfo
   ```

## Step 1: Transfer the SLURM script to Stampede3

Transfer `run_contact_rich_training.slurm` to Stampede3 using `scp` or `rsync` from your local machine. Do not use `$HOME` (15 GB limit) — put it in `$WORK` or `$SCRATCH`.

```bash
scp run_contact_rich_training.slurm <username>@stampede3.tacc.utexas.edu:$SCRATCH/
```

## Step 2: Submit the training job

```bash
sbatch run_contact_rich_training.slurm
```

```bash
# Check job status
squeue -u $USER
```

## Step 3: Monitor training

```bash
# Follow live output (replace <jobid> with your SLURM job ID)
tail -f gear_assembly_train.<jobid>.out
```

Isaac Lab prints per-iteration stats as training progresses. You should see `Mean reward` trending upward over time. Training runs for 1500 iterations and takes approximately **24 hours** on this configuration.

## Step 4: Locate outputs

After the job finishes, all artifacts are under:

```
$SCRATCH/isaaclab_runs/gear_assembly_<jobid>/results/logs/rsl_rl/gear_assembly_ur10e/<timestamp>/
├── model_*.pt          # policy checkpoints (saved periodically)
├── params/             # run config snapshot
└── videos/
    └── train/          # MP4s recorded every 5000 steps
```

```bash
# List checkpoints
ls -lh $SCRATCH/isaaclab_runs/gear_assembly_<jobid>/results/logs/rsl_rl/gear_assembly_ur10e/*/model_*.pt

# List training videos
ls -lh $SCRATCH/isaaclab_runs/gear_assembly_<jobid>/results/logs/rsl_rl/gear_assembly_ur10e/*/videos/train/
```

> **Best practice — copy to `$WORK` before purge.**  
> `$SCRATCH` files inactive for 10 days may be deleted. Copy your final checkpoint and videos to `$WORK` as soon as training finishes:
> ```bash
> cp -r $SCRATCH/isaaclab_runs/gear_assembly_<jobid>/results/logs $WORK/gear_assembly_logs_<jobid>
> ```

## How 8-GPU distributed training works

Uses `isaaclab.sh -p -m torch.distributed.run` to spawn one process per GPU inside the container's Python environment via PyTorch DDP:

| Setting | Value |
|---------|-------|
| GPUs | 8× RTX PRO 6000 Blackwell (48 GB each) |
| Environments per GPU | 256 |
| **Total environments** | **2048** |
| Launch command | `isaaclab.sh -p -m torch.distributed.run --nproc_per_node=8` |

Each GPU runs 256 independent parallel simulation environments. The policy network gradients are synchronized across GPUs via DDP at each update step. Note that for this contact-rich task the physics simulation (collection) dominates iteration time, so wall-clock speedup over single-GPU is limited.

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `run_contact_rich_training.slurm` | Ready-to-submit SLURM batch script for 8-GPU full training |
