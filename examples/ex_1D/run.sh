#!/bin/bash

if [ $# -eq 0 ]; then
	mode="train"
else
	mode=$1
fi


seed=10
#################
epochs=3000
lr=1.e-2
#################
nodes=10
codim=1
#################
datasize=51
batch=-1
#################
power_iters=1
w_decay=0
#################
alpha_TV=1.e-1
alpha_model=0
alpha_rhsjac=0.0
alpha_rhsdiv=1.e-1
alpha_fpdiv=1.e-1
#################
T=5
steps=5
#################

for theta in 0.0 0.5 1.0; do
python3 ex_1D.py --seed $seed --epochs $epochs --theta $theta --lr $lr --T $T --steps $steps --datasize $datasize --batch $batch --nodes $nodes --codim $codim --mode $mode --init rnd --w_decay $w_decay --alpha_rhsjac $alpha_rhsjac --alpha_rhsdiv $alpha_rhsdiv --alpha_TV $alpha_TV --alpha_model $alpha_model --alpha_fpdiv $alpha_fpdiv --power_iters $power_iters
done