#!/bin/bash


# seed=10
# #################
# # resnet params
# theta=0.0
# tol=1.e-6
# T=2.0
# steps=20
# #################
# # rhs params
# codim=0
# width=4
# depth=2
# sigma=relu
# # rhs spectral properties
# scales=learn
# piters=1
# eigs='-25 -15'
# #################
# # training params
# init=rnd
# epochs=50
# lr=1.e-2
# datasize=100
# datasteps=2
# batch=1
# #################
# # regularizer params
# wdecay=0
# aTV=1.e-1
# ajdiag=0.e0
# diaval=-1.0
# adiv=0.e0
# ajac=0.e-4
# af=0.e-3
# aresid=0.e0
# mciters=1
# #################
# prefix=mlp


if [ $# -eq 0 ]; then
	mode="train"
else
	mode=$1
	# adiv=$2
	# eig=$2
	# ajdiag=$2
	# prefix=$2
	# theta=$2
	datasteps=$2
fi



function run {
	python ex_stiff.py   \
		--mode     $mode \
		\
		--seed     10    \
		\
		--theta    $1    \
		--tol      1.e-6 \
		--T        2     \
		--steps    20    \
		\
		--codim    0     \
		--width    4     \
		--depth    2     \
		--sigma    relu  \
		\
		--scales   learn \
		--piters   1     \
		--eigs   -25 -15 \
		\
		--init     rnd   \
		--epochs   50    \
		--lr       1.e-2 \
		--datasize 100   \
		--datasteps $2   \
		--batch     1    \
		\
		--aTV      1.e-1 \
		--mciters  1
}



# function run {
# 	python ex_stiff.py \
# 		--prefix $prefix \
# 		--mode   $1      --seed   $seed \
# 		--theta  $2      --tol    $tol    --T     $T     --steps    $steps \
# 		--codim  $codim  --width  $width  --depth $depth --sigma    $sigma \
# 		--scales $scales --piters $piters --eigs  $eigs \
# 		--init   $init   --epochs $epochs --lr    $lr    --datasize $datasize --datasteps $datasteps --batch  $batch \
# 		--wdecay $wdecay --aTV    $aTV    --adiv  $adiv  --ajac     $ajac     --af        $af        --aresid $aresid --mciters $mciters
# }


# for datasteps in 1 2 4 10 20; do
# 	for theta in 0.0 0.25 0.5 0.75 1.0; do
# for datasteps in 1; do
	for theta in 0.0 0.5 1.0; do
		run $theta $datasteps
	done
# done






