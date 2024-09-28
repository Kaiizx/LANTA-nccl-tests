#!/bin/bash
#SBATCH -p gpu --exclusive               # Specify partition [Compute/Memory/GPU]
#SBATCH -c 64                            # Specify number of processors per task
#SBATCH -N 32
#SBATCH -w lanta-g-[032-063]
#SBATCH --ntasks-per-node=1		         # Specify number of tasks per node
#SBATCH --gpus-per-node=4		         # Specify total number of GPUs
#SBATCH -t 1:00:00                       # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt999001                      # Specify project name
#SBATCH -J nccl-test                       # Specify job name
#SBATCH -o nccl-%j.out        # Specify output file

module load PrgEnv-gnu
module load cpe-cuda/23.03
# module load cray-mpich
module load cudatoolkit/23.3_11.8
# module load OpenMPI
# modele load nccl


# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

export NCCL_TIMEOUT=3600000
export NCCL_SOCKET_IFNAME=hsn
#export NCCL_SOCKET_NTHREADS=8
#export NCCL_NSOCKS_PERTHREAD=2

# export MPI_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/openmpi/openmpi-3.1.5/include/openmpi/ompi/mpi 
# export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda 
# export NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/11.8/nccl

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/11.8/nccl/lib/:$LD_LIBRARY_PATH


echo $LD_LIBRARY_PATH

echo $CRAY_LD_LIBRARY_PATH
# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

# make MPI=1 MPI_HOME=/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1 NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/11.8/nccl 

srun /project/lt999001-intern/atikan/code/nccl-tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 4
