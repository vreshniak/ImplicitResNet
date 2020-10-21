#!/bin/bash

if [ $# -eq 0 ]; then
	mode="train"
else
	mode=$1
	# adiv=$2
fi


seed=10
#################
# resnet params
method=inner
theta=0.0
tol=1.e-6
T=5
#################
# rhs params
codim=1
width=10
depth=3
sigma=gelu
# rhs spectral properties
scales=equal
piters=0
minrho=1.e-2
maxrho=3.0
#################
# training param
init=rnd
epochs=3000
lr=1.e-3
datasize=20
batch=-1
#################
wdecay=0
aTV=-1.0
# adiv=1.e0
ajac=1.e-1
af=0.e-3
atan=0.0
aresid=1.0
#################

for tol in 1.e-6; do
	for datasize in 20; do
		for T in 5; do
			for adiv in 0.0 0.25 0.5 0.75 1.0; do
				for theta in 0.0 0.25 0.5 0.75 1.0; do
					python ex_1D.py \
					--mode $mode --seed $seed \
					--method $method --theta $theta --tol $tol --T $T \
					--codim $codim --width $width --depth $depth --sigma $sigma \
					--scales $scales --piters $piters --minrho $minrho --maxrho $maxrho \
					--init $init --epochs $epochs --lr $lr --datasize $datasize --batch $batch \
					--wdecay $wdecay --aTV $aTV --adiv $adiv --ajac $ajac --atan $atan --af $af --aresid $aresid
				done
			done
		done
	done
done