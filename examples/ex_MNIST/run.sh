#!/bin/bash


# modes=( 'train' 'test' 'process' )
# modes=( 'test' 'process' )
# modes=( 'process' )
modes=( 'train' )

# hyperparameters search
for dataset in MNIST CIFAR10; do
	for augm in clean; do
		for data in 100; do
			for batch in 100; do
				for T in 1 3; do
					for theta in 0.00 0.25 0.50 0.75 1.00; do
						for adiv in 5.00; do
							for center in 1.00 0.75 0.50 0.25 0.00; do
								for mode in "${modes[@]}"; do
									python -W ignore ex_MNIST.py --rhs_mode center --name cnt_"$center"_"$augm"_data_"$data"_batch_"$batch"_T_"$T"_adiv_"$adiv"_theta_"$theta" --dataset $dataset --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --stabcenter 1.0 $center --batch $batch --relaxation 100
								done
							done
						done
					done
				done
			done
		done
	done
done





# # hyperparameters search
# for dataset in MNIST CIFAR10; do
# 	for augm in clean; do
# 		for data in 1000; do
# 			for batch in 100; do
# 				for T in 1 3; do
# 					for adiv in 1.00 5.00; do
# 						for theta in 0.00 0.25 0.50 0.75 1.00; do
# 						# for theta in 0.50 0.75; do
# 							# case $theta in
# 							# 	0.00 | 0.25)
# 							# 		lims=( -1.0 -0.8 -0.6 -0.4 -0.2 0.0 0.2 0.4 )
# 							# 		;;

# 							# 	0.50)
# 							# 		lims=( -0.33 ) #( -0.8 -0.6 -0.4 -0.2 0.0 0.2 0.4 )
# 							# 		;;

# 							# 	0.75)
# 							# 		lims=( -0.14 ) #( -0.4 -0.2 0.0 0.2 0.4 )
# 							# 		;;

# 							# 	1.00)
# 							# 		lims=( 0.0 0.2 0.4 )
# 							# 		;;
# 							# esac
# 							for

# 							for lim in "${lims[@]}"; do
# 								for mode in "${modes[@]}"; do
# 									python -W ignore ex_MNIST.py --name lim_"$lim"_"$augm"_data_"$data"_batch_"$batch"_T_"$T"_adiv_"$adiv"_theta_"$theta" --dataset $dataset --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --stablim $lim 1.0 1.1 --learn_limits --batch $batch --relaxation 50
# 									# case $mode in
# 									# 	'test' | 'process')
# 									# 	for relax in 0 10 20 30 40 50; do
# 									# 		python -W ignore ex_MNIST.py --name lim_"$lim"_"$augm"_data_"$data"_batch_"$batch"_T_"$T"_adiv_"$adiv"_theta_"$theta"_relax_"$relax" --dataset $dataset --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --stablim $lim 1.0 1.1 --learn_limits --batch $batch
# 									# 	done
# 									# 	;;
# 									# esac
# 								done
# 								# for relax in 0 10 20 30 40 50; do
# 								# 	for mode in 'test' 'process'; do
# 								# 		python -W ignore ex_MNIST.py --name lim_"$lim"_"$augm"_data_"$data"_batch_"$batch"_T_"$T"_adiv_"$adiv"_theta_"$theta"_relax_"$relax" --dataset $dataset --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --stablim $lim 1.0 1.1 --learn_limits --batch $batch
# 								# 	done
# 								# done
# 							done
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done



# # initialization search
# for augm in clean; do
# 	for data in 1000; do
# 		for batch in 100; do
# 			for T in 5; do
# 				for adiv in 0.00; do
# 					for theta in 0.00; do
# 						for ini in 1.0 0.5 0.2 0.1 0.0 -0.1 -0.2 -0.5 -1.0; do
# 							for mode in train test; do
# 								python -W ignore ex_MNIST.py --name "$augm"_data_"$data"_batch_"$batch"_T_"$T"_adiv_"$adiv"_theta_"$theta"_ini_"$ini" --model 1 --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --stablim -2.0 $ini 2.0 --learn_shift --batch $batch
# 							done
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done



# # model search
# for augm in clean; do
# 	for data in 1000; do
# 		for batch in 100; do
# 			for T in 1; do
# 				for adiv in 1.00; do
# 					for theta in  0.00; do
# 						for model in 123; do
# 							for mode in "${modes[@]}"; do
# 								python -W ignore ex_MNIST.py --name model_"$model"_"$augm"_data_"$data"_batch_"$batch"_T_"$T"_adiv_"$adiv"_theta_"$theta" --model $model --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --dissipative --stabzero --learn_shift --batch $batch
# 							done
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done



# for augm in clean; do
# 	for data in 1000; do
# 		for batch in 100; do
# 			for T in 1 3; do
# 				# for theta in 0.00; do
# 				# 	for mode in  test; do
# 				# 		python -W ignore ex_MNIST.py --name "$augm"_data_"$data"_batch_"$batch"_T_"$T"_plain_theta_"$theta" --model 1 --datasize $data --mode $mode --T $T --theta $theta --adiv 0.0 --piters 0 --batch $batch
# 				# 	done
# 				# done

# 				# for adiv in 0.00; do
# 				# 	for theta in 0.00; do
# 				# 		for mode in  test; do
# 				# 			python -W ignore ex_MNIST.py --name "$augm"_data_"$data"_batch_"$batch"_T_"$T"_adiv_"$adiv"_theta_"$theta"_1Lip --model 1 --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --eiglim -1.0 0.0 1.0 --batch $batch --datanoise $augm
# 				# 		done
# 				# 	done
# 				# done

# 				# for adiv in 0.00; do
# 				# 	for theta in 0.00; do
# 				# 		for mode in  test; do
# 				# 			python -W ignore ex_MNIST.py --name "$augm"_data_"$data"_batch_"$batch"_T_"$T"_adiv_"$adiv"_theta_"$theta" --model 1 --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --stablim -1.0 1.0 2.0 --learn_shift --batch $batch --datanoise $augm
# 				# 		done
# 				# 	done
# 				# done

# 				for adiv in 1.00; do
# 					for theta in 0.00 0.25 0.50 0.75; do
# 						for mode in  test; do
# 							python -W ignore ex_MNIST.py --name "$augm"_data_"$data"_batch_"$batch"_T_"$T"_adiv_"$adiv"_theta_"$theta"_minstab_0.0 --model 123 --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --stablim -1.0 1.0 2.0 --learn_shift --batch $batch --minstab 0.0 --datanoise $augm
# 						done
# 					done
# 				done

# 			done
# 		done
# 	done
# done



# for augm in clean; do
# 	for data in -1; do
# 		for T in 1; do
# 			# for theta in 0.25 0.50 0.75 1.00; do
# 			# 	for mode in train test; do
# 			# 		python -W ignore ex_MNIST.py --name "$augm"_data_"$data"_T_"$T"_plain_theta_"$theta" --model 1 --datasize $data --mode $mode --T $T --theta $theta --adiv 0.0 --piters 0 --batch 256 --lr 1.e-3
# 			# 	done
# 			# done

# 			# for theta in 0.25 0.50 0.75 1.00; do
# 			# 	for mode in train test; do
# 			# 		python -W ignore ex_MNIST.py --name "$augm"_data_"$data"_T_"$T"_1Lip_theta_"$theta" --model 1 --datasize $data --mode $mode --T $T --theta $theta --adiv 0.0 --piters 1 --eiglim -1.0 0.0 1.0 --batch 256 --lr 1.e-3
# 			# 	done
# 			# done

# 			for adiv in 1.00; do
# 				for theta in 0.00 0.25 0.50; do
# 					for lim in -1.0 -0.8 -0.6 -0.4 -0.2 0.0 0.2 0.4; do
# 					# for lim in -0.8 -0.4 0.0; do
# 						for mode in  test; do
# 							python -W ignore ex_MNIST.py --name "$augm"_data_"$data"_T_"$T"_adiv_"$adiv"_theta_"$theta"_lim_"$lim" --model 1 --datasize $data --mode $mode --T $T --theta $theta --adiv $adiv --piters 1 --stablim $lim 1.0 2.0 --learn_shift --batch 500 --lr 1.e-2
# 						done
# 					done
# 				done
# 			done
# 		done
# 	done
# done



# python make_table.py
# pdflatex --extra-mem-bot=10000000 results.tex
# rm *.aux *.log *.bak
# open results.pdf
