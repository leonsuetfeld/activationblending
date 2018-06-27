#!/bin/bash
#$ -N m_ASC_main                                                                  # change when swapping between deployed and development folders
#$ -wd /net/store/ni/users/lsuetfel/activations_grid/                           # change when swapping between deployed and development folders
##$ -cwd
#$ -l h_rt=01:29:59
#$ -l mem=6G
#$ -l mem_free=6G
#$ -l nv_mem_free=2900M
#$ -l ubuntu_version=xenial
#$ -l cuda=1
#$ -l cuda_capability=500
#$ -l cuda_driver=8000.000000
#$ -l cuda_cores=640
##$ -l h=*cippy*
#$ -l h=!*picture*&!*isonoe*&!*kalyke*
#$ -t 1:1
#$ -p 0 ## priority, only negative integers allowed
##$ -cwd
#$ -j y
#$ -o /net/store/ni/users/lsuetfel/activations_grid/3_gridjob_outfiles/         # change when swapping between deployed and development folders

# definition of commands for later execution
UVENV="source venvtfgpu/bin/activate"
UAPP="python3"
UCWD="/net/store/ni/users/lsuetfel/activations_grid/2_scheduling/"              # change when swapping between deployed and development folders
UMCR="m_advanced_scheduler.py"

# write header for return files
echo "*** Start of job ***"
date
echo ""
echo "Hostname"
echo "$HOSTNAME"
echo ""
echo "Command:"
echo "$UVENV"
echo "$UCUDA"
echo "$UAPP $UCWD$UMCR $SGE_TASK_ID"
echo "deactivate"
echo ""
echo "Start"

# execute code using the predefined commands (see top)
# export THEANO_FLAGS="base_compiledir=$TMPDIR" # check tensorflow flag to ensure that multiple jobs dont use the same temp file/ temp location

$UVENV

export THEANO_FLAGS=floatX=float32,device=gpu0,lib.cnmem=0.8
export CUDNN_HOME=/net/store/ni/users/lsuetfel/activations_grid/cuda
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDNN_HOME/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDNN_HOME/lib64"

$UAPP $UCWD$UMCR $SGE_TASK_ID
deactivate

# write footer for return files
echo ""
date
echo "*** End of job ***"
