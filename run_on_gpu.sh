#!/usr/bin/env bash
# Usage:
#   ./run_on_gpu.sh <script.py> [args...]   -> quick test, chay truc tiep tren login node (khong qua GPU/Slurm)
#   ./run_on_gpu.sh --train <script.py>     -> submit training/compute lon qua Slurm (sbatch) tren A100
#   ./run_on_gpu.sh --status                -> xem squeue (job dang cho/chay)
#   ./run_on_gpu.sh --log [jobid]           -> xem slurm-<jobid>.out (mac dinh: file moi nhat)
#   ./run_on_gpu.sh --cancel <jobid>        -> scancel 1 job
#
# Luu y: KHONG bao gio sua file run_via_slurm. File `script` chi duoc sua dong 12 (ten file python).
set -e

MUTAGEN="mutagen.exe"
REMOTE_HOST="a100-B"
REMOTE_DIR="~/thuongnm_hust/TestSSH"
CONDA_ENV="thuongnm_gpu"          # Python 3.11 + torch cu124, dung chung cho ca quick test lan GPU/Slurm
CONDA_SH="/data2/shared/apps/conda/etc/profile.d/conda.sh"

MODE="quick"
case "$1" in
  --train)  MODE="train"; shift ;;
  --status) MODE="status"; shift ;;
  --log)    MODE="log"; shift ;;
  --cancel) MODE="cancel"; shift ;;
esac

"$MUTAGEN" sync flush testssh >/dev/null

case "$MODE" in
  quick)
    SCRIPT="$1"
    shift || true
    ssh "$REMOTE_HOST" "source $CONDA_SH && conda activate $CONDA_ENV && cd $REMOTE_DIR && python $SCRIPT $*"
    ;;

  train)
    SCRIPT="$1"
    REMOTE_CMD="cd $REMOTE_DIR && sed -i '12s#.*#python $SCRIPT#' script && conda activate $CONDA_ENV && . run_via_slurm && sleep 2 && squeue -u \$USER"
    ssh "$REMOTE_HOST" "bash -lic \"$REMOTE_CMD\""
    ;;

  status)
    REMOTE_CMD="squeue -u \$USER"
    ssh "$REMOTE_HOST" "bash -lic \"$REMOTE_CMD\""
    ;;

  log)
    JOBID="$1"
    if [ -z "$JOBID" ]; then
      ssh "$REMOTE_HOST" "cd $REMOTE_DIR && ls -t slurm-*.out 2>/dev/null | head -1 | xargs cat"
    else
      ssh "$REMOTE_HOST" "cat $REMOTE_DIR/slurm-$JOBID.out"
    fi
    ;;

  cancel)
    JOBID="$1"
    REMOTE_CMD="scancel $JOBID"
    ssh "$REMOTE_HOST" "bash -lic \"$REMOTE_CMD\""
    ;;
esac
