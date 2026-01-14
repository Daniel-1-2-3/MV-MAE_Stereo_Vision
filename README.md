# Building the sif container #
python3 -m venv .whvenv
source .whvenv/bin/activate
python -m pip install -U pip

rm -rf wheelhouse
mkdir -p wheelhouse

python -m pip download \
  --only-binary=:all: --prefer-binary \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --abi cp312 \
  --python-version 312 \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r requirements.txt \
  -d wheelhouse

deactivate

ls -lh wheelhouse | head
SCRATCH=/scratch/$USER/apptainer_build
mkdir -p "$SCRATCH/cache" "$SCRATCH/tmp"

export APPTAINER_CACHEDIR="$SCRATCH/cache"
export APPTAINER_TMPDIR="$SCRATCH/tmp"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

apptainer cache clean --type=all --force || true
apptainer build training.sif training.def

# Start training, see terminal output for errors #
chmod +x training.sh
sbatch training.sh

# Important files and folders #
- The MAE_Model/ folder contains MV-MAE files. The file containing the entire model is model.py
- The Mujoco_Sim/ folder contains Mujoco environment files. The pick_env.py is the environment with step() and get_obs(), reset(), etc functions. The brax_wrapper.py is wraps the env for training with Brax. 
- train_drqv2_mujoco.py is the training script. 

