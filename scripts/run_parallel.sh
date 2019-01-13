#!/usr/bin/env zsh

### Job name
#BSUB -J "parallel_agent"
 
### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#BSUB -o parallel_agent.%J.%I
 
### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 00:03
 
### Request memory you need for your job in TOTAL in MB
#BSUB -M 12288

### Request a gpu machine
#BSUB -gpu -
###BSUB -a gpu Joel said this is deprecated
#BSUB -P um_dke
###BSUB -R kepler


module switch intel gcc
module load python/3.6.0
module load cuda/90
module load cudnn/7.0.5

cd /home/hn217262/RLaSpa

### only needed at first time running -->
#pip3 install --user tensorflow
#pip3 install --user https://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
#pip3 install --user gym
### <-- only needed at first time running 

###PYTHONPATH=$PYTHONPATH:/home/hn217262/RLaSpa/src 

export PYTHONPATH="${PYTHONPATH}:/home/hn217262/RLaSpa/src:/home/hn217262/RLaSpa"
echo $PYTHONPATH
pwd
python3 src/agents/parallel.py
