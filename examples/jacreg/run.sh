#!/bin/bash

if [ $# -eq 0 ]; then
	mode="train"
else
	mode=$1
fi


seed=10
#################
epochs=10000
lr=1.e-2
#################
nodes=10
#################
datasize=11
batch=-1
#################
w_decay=0
#################
alpha_jac=0.1
#################

python3 MLP.py --seed $seed --mode $mode --epochs $epochs --lr $lr --datasize $datasize --batch $batch --nodes $nodes --w_decay $w_decay --alpha_jac $alpha_jac