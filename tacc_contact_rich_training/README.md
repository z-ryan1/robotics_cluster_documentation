# Contact-Rich Gear Assembly Training on Stampede3 (8-GPU)

Run the full Isaac Lab RL training for the contact-rich gear assembly task using all 8 RTX PRO 6000 Blackwell GPUs on a single `amd-rtx` node.

## Prerequisites

1. **Isaac Lab container pulled** — follow the [Isaac Lab NGC Container guide](../tacc_stampede_isaaclab_container/) to pull `nvcr.io/nvidia/isaac-lab:2.3.2` to `$SCRATCH/containers/isaaclab/isaac-lab_2.3.2.sif`.

2. **`amd-rtx` allocation** — confirm you have an active allocation:

   ```bash
   /usr/local/etc/taccinfo
   ```

## Step 1: Clone the task repo to `$WORK`

Clone once from a login node. `$WORK` persists across login and compute nodes.

```bash
cd $WORK
git clone <contact_rich_robot_manipulation_skills_repo_url> contact_rich
```

The repo is documentation-based — the actual task environment is registered in the Isaac Lab container. No additional install is needed inside the container; the container's `/workspace/isaaclab` already includes the gear assembly task under `isaaclab_tasks`.

> If the gear assembly task is **not** bundled in the NGC container image, you would need to bind-mount the local task source and run `pip install -e` inside the container before training. Check that `Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0` appears in the task registry first (see Verify section below).

## Step 2: Submit the training job

```bash
cd $WORK/contact_rich   # or wherever you cloned this docs repo
sbatch tacc_contact_rich_training/run_contact_rich_training.slurm
```

This submits a 24-hour job on 1 node using all 8 GPUs. Expected training time with 8 GPUs is roughly **3–6 hours** compared to 12–24 hours on a single GPU.

```bash
# Check job status
squeue -u $USER
```

## Step 3: Monitor training

### Watch the log

```bash
# Follow live output (replace <jobid> with your SLURM job ID)
tail -f gear_assembly_train.<jobid>.out
```

Isaac Lab prints per-iteration reward stats as training progresses. You should see `mean_reward` trending upward after the first few thousand iterations.

### TensorBoard from a login node

Isaac Lab writes TensorBoard event files alongside the checkpoints. To view them:

```bash
# On the login node, find your run directory
RUN_DIR="$SCRATCH/isaaclab_runs/gear_assembly_<jobid>"

# Forward TensorBoard to your local machine (run this on your laptop)
ssh -L 6006:localhost:6006 <username>@stampede3.tacc.utexas.edu \
  "module load python3; pip install tensorboard -q; tensorboard --logdir $RUN_DIR/results/logs --port 6006"
```

Then open `http://localhost:6006` in your browser.

## Step 4: Locate outputs

After the job finishes, all artifacts are under:

```
$SCRATCH/isaaclab_runs/gear_assembly_<jobid>/results/logs/rsl_rl/gear_assembly/<timestamp>/
├── model_*.pt          # policy checkpoints (saved periodically)
├── params/             # run config snapshot
└── videos/
    └── train/          # MP4s recorded every 5000 steps
```

```bash
# List checkpoints
ls -lh $SCRATCH/isaaclab_runs/gear_assembly_<jobid>/results/logs/rsl_rl/gear_assembly/*/model_*.pt

# List training videos
ls -lh $SCRATCH/isaaclab_runs/gear_assembly_<jobid>/results/logs/rsl_rl/gear_assembly/*/videos/train/
```

> **Best practice — copy to `$WORK` before purge.**  
> `$SCRATCH` files inactive for 10 days may be deleted. Copy your final checkpoint and videos to `$WORK` as soon as training finishes:
> ```bash
> cp -r $SCRATCH/isaaclab_runs/gear_assembly_<jobid>/results/logs $WORK/gear_assembly_logs_<jobid>
> ```

## Verify the task is registered (optional pre-flight check)

Before submitting the 24-hour job, confirm the task name is recognized by the container using a short interactive test:

```bash
idev -p amd-rtx -N 1 -n 1 -t 00:30:00

# Inside the compute node:
module reset && module load nvidia && module load tacc-apptainer/1.4.1
ISAACLAB_SIF="$SCRATCH/containers/isaaclab/isaac-lab_2.3.2.sif"

TERM=xterm apptainer exec --cleanenv --fakeroot --nv "$ISAACLAB_SIF" \
  bash --noprofile --norc -lc '
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
/workspace/isaaclab/isaaclab.sh -p -c "
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg(\"Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0\", num_envs=1)
print(\"Task found:\", cfg)
"
'
```

If the task is not found, you'll need to bind-mount and install the gear assembly task source before training — open an issue or check the contact-rich repo's `README.md` for install instructions.

## How 8-GPU distributed training works

The SLURM script passes `--distributed` to the Isaac Lab training script. This enables **PyTorch DDP (Distributed Data Parallel)** across all GPUs visible inside the Apptainer container:

| Setting | Value |
|---------|-------|
| GPUs | 8× RTX PRO 6000 Blackwell (48 GB each) |
| Environments per GPU | 256 |
| **Total environments** | **2048** |
| Training flag | `--num_envs 2048 --distributed` |

Each GPU runs 256 independent parallel simulation environments. The policy network gradients are synchronized across GPUs via DDP at each update step, giving the same convergence as a single-GPU run but roughly 8× faster wall time.

> **Note:** If your Isaac Lab version does not support `--distributed` (check with `isaaclab.sh -p train.py --help`), replace it with:
> ```bash
> torchrun --nnodes=1 --nproc_per_node=8 \
>   /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
>   --task Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0 \
>   --headless --num_envs 2048 --video --video_length 800 --video_interval 5000
> ```

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `run_contact_rich_training.slurm` | Ready-to-submit SLURM batch script for 8-GPU full training |
