#!/bin/bash
#$ -N SBF_9e
#$ -wd /net/store/ni/users/lsuetfel/activations/
##$ -cwd
#$ -l h_rt=01:29:00
#$ -l mem=3G
#$ -l mem_free=3G
#$ -l nv_mem_free=2800M
#$ -l ubuntu_version=xenial
#$ -l cuda=1
#$ -l cuda_capability=500
#$ -l cuda_driver=8000.000000
#$ -l cuda_cores=760
##$ -l h=*cippy*
#$ -l h=!*picture*
#$ -t 1:40
#$ -p 0 ## priority, only negative integers allowed
##$ -cwd
#$ -j y
#$ -o /net/store/ni/users/lsuetfel/activations/3_gridjob_output_files/

# definition of commands for later execution
UVENV="source venvtfgpu/bin/activate"
UAPP="python3"
UCWD="/net/store/ni/users/lsuetfel/activations/"
UMCR="deepnet_supervisor_SBF_9e.py"

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
export CUDNN_HOME=/net/store/ni/users/lsuetfel/activations/cuda
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDNN_HOME/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDNN_HOME/lib64"

$UAPP $UCWD$UMCR $SGE_TASK_ID
deactivate

# write footer for return files
echo ""
date
echo "*** End of job ***"
