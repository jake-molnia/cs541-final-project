#!/bin/bash
#SBATCH --job-name=email_classification
#SBATCH --output=slurm_output_%j.log
#SBATCH --error=slurm_error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=academic
#SBATCH --gres=gpu:1

# Print job details
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $SLURM_GPUS"

# Load necessary modules for GPU support
module purge
module load cudnn8.9-cuda12.3/8.9.7.29

# Navigate to the project directory
cd $HOME/scratch/01-label-data

# Set up Python environment using uv (which is in the current directory)
echo "Setting up UV environment..."
if [ ! -d ".venv" ]; then
    echo "Creating new UV environment..."
    ./uv venv .venv
    ./uv pip install -r requirements.txt
else
    echo "Using existing UV environment..."
fi

# Activate the UV environment
source .venv/bin/activate

# Run the script
echo "Starting classification script..."
python diyar.py

echo "Job completed at $(date)"
