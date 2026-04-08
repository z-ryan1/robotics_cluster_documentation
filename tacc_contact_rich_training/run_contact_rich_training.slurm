#!/bin/bash
#SBATCH -J gear_assembly_train
#SBATCH -p amd-rtx
#SBATCH -N 1                           # 1 node
#SBATCH -n 1                           # 1 task (Isaac Lab handles multi-GPU internally)
#SBATCH -t 24:00:00                    # 24-hour wall time for full training
#SBATCH -o gear_assembly_train.%j.out
#SBATCH -e gear_assembly_train.%j.err

# Full training run for Isaac Lab gear assembly contact-rich manipulation task.
# Uses all 8 RTX PRO 6000 Blackwell GPUs on a single amd-rtx node via
# PyTorch DDP (torch.distributed.run). 2048 parallel environments total
# (256 per GPU).
#
# Prerequisites:
#   - Isaac Lab container pulled to $SCRATCH/containers/isaaclab/isaac-lab_2.3.2.sif
#     (see tacc_stampede_isaaclab_container/ guide)
#
# Submit:
#   sbatch run_contact_rich_training.slurm

set -euo pipefail

# Some modulefiles/source'd completion scripts are not nounset-clean.
set +u
module reset
module load nvidia
module load tacc-apptainer/1.4.1
set -u

# ── Container image ───────────────────────────────────────────────────────────
export ISAACLAB_IMG_DIR="$SCRATCH/containers/isaaclab"
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer/cache"
export APPTAINER_TMPDIR="$SCRATCH/.apptainer/tmp"
mkdir -p "$ISAACLAB_IMG_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

ISAACLAB_SIF="$ISAACLAB_IMG_DIR/isaac-lab_2.3.2.sif"
if [[ ! -f "$ISAACLAB_SIF" ]]; then
  echo "ERROR: Isaac Lab container not found at $ISAACLAB_SIF"
  echo "Pull it first following the tacc_stampede_isaaclab_container guide."
  exit 1
fi

# ── Run directory (on $SCRATCH for fast I/O) ──────────────────────────────────
RUN_DIR="$SCRATCH/isaaclab_runs/gear_assembly_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/kit-cache" "$RUN_DIR/kit-data" "$RUN_DIR/isaaclab-tmp"

echo "============================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Output dir:  $RUN_DIR"
echo "============================================================"

# ── Launch training inside container ─────────────────────────────────────────
# Uses isaaclab.sh -p -m torch.distributed.run to spawn one process per GPU
# inside the container's Python environment. Each process runs 256 environments.
# Total: 2048 environments across 8 GPUs.
TERM=xterm apptainer exec --cleanenv --fakeroot --nv \
  --bind "$RUN_DIR/results:/results,$RUN_DIR/kit-cache:/isaac-sim/kit/cache,$RUN_DIR/kit-data:/isaac-sim/kit/data,$RUN_DIR/isaaclab-tmp:/tmp/isaaclab" \
  --pwd /results \
  "$ISAACLAB_SIF" \
  bash --noprofile --norc -lc '
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y

/workspace/isaaclab/isaaclab.sh -p -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=8 \
  /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0 \
  --headless \
  --num_envs 256 \
  --distributed \
  --video \
  --video_length 800 \
  --video_interval 5000
'

echo "============================================================"
echo "Training complete. Artifacts at: $RUN_DIR/results"
echo "Checkpoints: $RUN_DIR/results/logs/rsl_rl/gear_assembly/<timestamp>/model_*.pt"
echo "Videos:      $RUN_DIR/results/logs/rsl_rl/gear_assembly/<timestamp>/videos/train/"
echo ""
echo "IMPORTANT: Copy checkpoints to \$WORK before they are purged from \$SCRATCH"
echo "  cp -r $RUN_DIR/results/logs \$WORK/gear_assembly_logs_${SLURM_JOB_ID}"
echo "============================================================"
