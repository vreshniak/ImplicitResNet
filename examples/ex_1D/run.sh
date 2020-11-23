#!/bin/bash

seed=10
#################
# resnet params
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
#################
# training param
init=rnd
epochs=3000
lr=1.e-3
datasize=20
batch=-1
#################
aTV=-1.0
ajac=1.e-1
mciters=1
#################



if [ $# -eq 0 ]; then
	mode="train"
else
	mode=$1
	# adiv=$2
fi



function run {
	python ex_1D.py \
		--mode   $mode   --seed   $seed \
		--theta  $1      --tol    $tol    --T     $T\
		--codim  $codim  --width  $width  --depth $depth  --sigma $sigma \
		--scales $scales --piters $piters \
		--init   $init   --epochs $epochs --lr    $lr    --datasize $datasize --batch  $batch \
		--aTV    $aTV    --adiv   $2      --ajac  $ajac  --mciters  $mciters
}



for adiv in 0.0 0.25 0.5 0.75 1.0; do
	for theta in 0.0 0.25 0.5 0.75 1.0; do
		run $theta $adiv
	done
done
