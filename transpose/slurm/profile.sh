#!/bin/bash

#SBATCH --job-name="mat_transpose_comp"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_user_prod
#SBATCH --error=mat_transpose_comp%j.err
#SBATCH --output=mat_transpose_comp%j.out
#SBATCH --account=ICT25_MHPC_0



module purge
module load gcc/12.2.0
module load nvhpc/24.5

nvc++ main.cu -o main.x

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running program..."
    # --- Execution ---
    ./main.x
else
    echo "Compilation failed."
    exit 1
fi