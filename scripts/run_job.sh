#!/usr/bin/env zsh

if [[ $# -eq 0 ]] ; then
    echo '>>>>>>>>>>>>>>>>>No task name is given!'
    exit 0
fi

RIGHT_NOW=$(date +"%Y-%m-%d_%H-%M-%S")

export TASK=${1}
export REPR=${2}

echo '>>>>>>>>>>>>>>>>>timestamp = \t' $RIGHT_NOW
echo '>>>>>>>>>>>>>>>>>task = \t' $TASK

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

export PYTHONPATH="${PYTHONPATH}:/home/hn217262/RLaSpa/src:/home/hn217262/RLaSpa"

SCRIPT_PATH="src/learning/final/experiment_final.py $TASK $REPR"
echo '>>>>>>>>>>>>>>>>>trying to run \t' $SCRIPT_PATH

python3 $SCRIPT_PATH
