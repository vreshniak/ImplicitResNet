import os
import sys
# import subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np
import pickle

import ex_setup

modes = ['train', 'test', 'process']
# modes = ['train']
# modes = ['test']


def hyperparameters_search():
	for dataset in ['CIFAR10']:
		acc_vs_cnt = {}
		for datasize in [1000]:
			for batch in [100]:
				for model in [3]:
					for rhs_mode in ['center','circle']:
						for T in [1]:
							for theta in [0.00, 0.25, 0.50, 0.75, 0.90, 1.00]:
								for center in [1.00, 0.75, 0.50, 0.25, 0.00]:
									for radius in [1.00]:
										for reg, reg_alpha in zip(['acnt','adiv'],[1.0,1.0]):
											for mode in modes:
												runname = "%s_%.2f_cnt_[1.00,%.2f]_rad_%.2f_model_%d_rhs_%s_data_%d_batch_%d_T_%d_theta_%.2f"%(reg,reg_alpha,center,radius,model,rhs_mode,datasize,batch,T,theta)
												options = "--name %s --mode %s --%s %.2f --model %d --dataset %s --datasize %d --batch %d --T %d --theta %.2f --stabcenter 1.0 %.2f --eigradius %.2f --rhs_mode %s --logdir hparam_search"%(runname,mode,reg,reg_alpha,model,dataset,datasize,batch,T,theta,center,radius,rhs_mode)
												# os.system("python -W ignore ex_MNIST_CIFAR.py %s"%(options))

											# load accuracy and spectrum
											with open(Path(dataset+'_results','output','name2accuracy'),'rb') as f: accuracy = pickle.load(f)
											with open(Path(dataset+'_results','output','name2spectrum'),'rb') as f: spectrum = pickle.load(f)

											# learned spectral circle
											c, r = spectrum[runname]['0.ode']['spectral_circle']
											c = ex_setup.theta_stability_fun(theta,c)

											# record learned stability centers
											key = "c_%s_%s_%d_%.2f"%(rhs_mode,reg,T,theta)
											acc_vs_cnt[key] = acc_vs_cnt[key]+[c] if key in acc_vs_cnt.keys() else [c]
											# record accuracy of trained model for various noise levels
											for eps in [0.0,0.05,0.1,0.2,0.3]:
												# top-1 accuracy
												acc = accuracy["train_GNclip_%.1f-%s"%(eps,runname)][0]
												key = "a_%s_%s_%d_%.2f_%.2f"%(rhs_mode,reg,T,theta,eps)
												acc_vs_cnt[key] = acc_vs_cnt[key]+[acc] if key in acc_vs_cnt.keys() else [acc]
		table = np.stack(list(acc_vs_cnt.values()), axis=1)
		header = " ".join(list(acc_vs_cnt.keys()))
		np.savetxt( Path(dataset+'_results','output','data','hparam_search.txt'), table, delimiter=' ', fmt='%.2f', header=header, comments='' )


def center_search():
	datasize = 1000
	batch    = 100

	epsilons = [0.0,0.1,0.2,0.3]
	# thetas   = [1.00, 0.75, 0.50, 0.25, 0.00]
	thetas   = [i*0.25 for i in range(5)]
	# thetas = [0.0]
	centers  = [i*0.1 for i in range(10,-1,-1)]  #[1.00, 0.75, 0.50, 0.25, 0.10, 0.05, 0.00]
	# centers = [1.0,0.5,0.1]
	times    = [1]
	for dataset in ['MNIST','FashionMNIST','CIFAR10']:
		reg_acc_vs_cnt = {}
		acc_vs_cnt     = {'c': centers}
		for model in [123]:
			for rhs_mode in ['stabcenter']:
				for T in times:
					for theta in thetas:
						for acnt in [0.00]:
							for radius in [0.00, 0.10, 1.00]:

								cntreg = 0.5 if radius==0.0 else 0.3

								# search over centers
								for center in centers:
									# eigcenter = max(ex_setup.theta_inv_stability_fun(theta, center), -20)
									# eigcntreg = max(ex_setup.theta_inv_stability_fun(theta, cntreg), -20)
									cnt    = "%.2f"%(center)    if acnt==0.00 else "[%.2f,%2f]"%(cntreg,center)
									# eigcnt = str(eigcenter) if acnt==0.00 else "%.2f %.2f"%(eigcntreg,eigcenter)
									for mode in modes:
										# runname = "cnt_%.2f_rad_%.2f_model_%d_rhs_%s_data_%d_batch_%d_T_%d_theta_%.2f"%(center,radius,model,rhs_mode,datasize,batch,T,theta)
										runname = "cnt_search_rad_%.2f_cnt_%s_theta_%.2f"%(radius,cnt,theta)
										options = "--name %s --mode %s --acnt %.2f --model %d --dataset %s --datasize %d --batch %d --T %d --theta %.2f --center %s --radius %.2f --rhs_mode %s --logdir center_search"%(runname,mode,acnt,model,dataset,datasize,batch,T,theta,cnt,radius,rhs_mode)
										# if not (acnt==1.00 and center>=cntreg):
										# 	os.system("python -W ignore ex_MNIST_CIFAR.py %s"%(options))

									# load accuracy
									with open(Path(dataset+'_results','output','name2accuracy'),'rb') as f: accuracy = pickle.load(f)

									# record accuracy of trained model for various noise levels
									acc_key = "train_GN_0.0-%s"%(runname)
									acc_clean = accuracy["train_GN_0.0-%s"%(runname)][0] if acc_key in accuracy.keys() else np.nan
									for eps in epsilons:
										acc_key = "train_GN_%.1f-%s"%(eps,runname)
										key = "a_%d_%.2f_%.2f_%.2f"%(T,radius,theta,eps)
										acc = accuracy[acc_key][0] if acc_key in accuracy.keys() and acc_clean==100 else np.nan
										acc_vs_cnt[key] = acc_vs_cnt[key]+[acc] if key in acc_vs_cnt.keys() else [acc]

							# # center regularization
							# for acnt in [1.00]:
							# 	for mode in modes:
							# 		runname = "cnt_[0.25,0.00]_rad_%.2f_acnt_%.2f_model_%d_rhs_%s_data_%d_batch_%d_T_%d_theta_%.2f"%(radius,acnt,model,rhs_mode,datasize,batch,T,theta)
							# 		options = "--acnt %.2f --name %s --mode %s --model %d --dataset %s --datasize %d --batch %d --T %d --theta %.2f --stabcenter 0.25 0.0 --eigradius %.2f --rhs_mode %s --logdir center_search"%(acnt,runname,mode,model,dataset,datasize,batch,T,theta,radius,rhs_mode)
							# 		os.system("python -W ignore ex_MNIST_CIFAR.py %s"%(options))

							# 	# load accuracy
							# 	with open(Path(dataset+'_results','output','name2accuracy'),'rb') as f: accuracy = pickle.load(f)

							# 	# record accuracy of trained model for various noise levels
							# 	for eps in epsilons:
							# 		acc_key = "train_GN_%.1f-%s"%(eps,runname)
							# 		key = "a_reg_%d_%.2f_%.2f"%(T,theta,eps)
							# 		acc = accuracy[acc_key][0] if acc_key in accuracy.keys() else np.nan
							# 		reg_acc_vs_cnt[key] = acc

		table = np.stack(list(acc_vs_cnt.values()), axis=1)
		header = " ".join(list(acc_vs_cnt.keys()))
		np.savetxt( Path(dataset+'_results','output','data','%s_center_search.txt'%(dataset)), table, delimiter=' ', fmt='%.2f', header=header, comments='' )

		# best_over_cnt = np.nanmax(table, axis=0)[1:]
		# reg_table = np.array(list(reg_acc_vs_cnt.values()))
		# table = np.hstack((np.array(epsilons), best_over_cnt, reg_table)).reshape(-1,len(epsilons)).T
		# header = " ".join(['eps']+['a_%d_%.2f'%(T,th) for T in times for th in thetas]+['a_reg_%d_%.2f'%(T,th) for T in times for th in thetas])
		# np.savetxt( Path(dataset+'_results','output','data','%s_center_search_vs_eps.txt'%(dataset)), table, delimiter=' ', fmt='%.2f', header=header, comments='' )


def radius_search():
	for dataset in ['MNIST','FashionMNIST','CIFAR10']:
		acc_vs_rad = {'r': [1.00, 0.75, 0.50, 0.25, 0.10]}
		for datasize in [1000]:
			for batch in [100]:
				for model in [123]:
					for rhs_mode in ['stabcircle']:
						for T in [1]:
							for theta in [1.00,0.75, 0.50, 0.25, 0.00]:
								for radius in [1.00, 0.75, 0.50, 0.25, 0.10]:
									for center in [0.00]:
										for acnt in [1.00]:
											for mode in modes:
												runname = "cnt_[%.2f,%.2f]_rad_%.2f_acnt_%.2f_model_%d_rhs_%s_data_%d_batch_%d_T_%d_theta_%.2f"%(radius,center,radius,acnt,model,rhs_mode,datasize,batch,T,theta)
												options = "--acnt %.2f --relaxation 0 --name %s --mode %s --model %d --dataset %s --datasize %d --batch %d --T %d --theta %.2f --stabcenter %.2f %.2f --eigradius %.2f --rhs_mode %s --logdir radius_search"%(acnt,runname,mode,model,dataset,datasize,batch,T,theta,radius,center,radius,rhs_mode)
												# os.system("python -W ignore ex_MNIST_CIFAR.py %s"%(options))

										# load accuracy
										with open(Path(dataset+'_results','output','name2accuracy'),'rb') as f: accuracy = pickle.load(f)

										# record accuracy of trained model for various noise levels
										for eps in [0.0,0.05,0.1,0.2,0.3]:
											# top-1 accuracy
											acc_key = "train_GNclip_%.1f-%s"%(eps,runname)
											key = "a_%d_%.2f_%.2f"%(T,theta,eps)
											acc = accuracy[acc_key][0] if acc_key in accuracy.keys() else np.nan
											acc_vs_rad[key] = acc_vs_rad[key]+[acc] if key in acc_vs_rad.keys() else [acc]
		table = np.stack(list(acc_vs_rad.values()), axis=1)
		header = " ".join(list(acc_vs_rad.keys()))
		np.savetxt( Path(dataset+'_results','output','data','%s_radius_search.txt'%(dataset)), table, delimiter=' ', fmt='%.2f', header=header, comments='' )



def search():
	for dataset in ['MNIST','FashionMNIST','CIFAR10']:
		acc_vs_eps = {'eps': [0.0, 0.1, 0.2, 0.3]}
		for datasize in [1000]:
			for batch in [100]:
				for model in [123]:
					for rhs_mode in ['stabcircle']:
						for T in [1]:
							for theta in [1.00, 0.75, 0.50, 0.25, 0.00]:
								for acnt, arad in zip([1.00],[0.1]):
									for mode in modes:
										runname = "cnt_[0.25,0]_rad_[1,0]_acnt_%.2f_arad_%.2f_model_%d_rhs_%s_data_%d_batch_%d_T_%d_theta_%.2f"%(acnt,arad,model,rhs_mode,datasize,batch,T,theta)
										options = "--acnt %.2f --arad %.2f --relaxation 0 --name %s --mode %s --model %d --dataset %s --datasize %d --batch %d --T %d --theta %.2f --stabcenter 0.25 0 --eigradius 1 0 --rhs_mode %s --logdir search"%(acnt,arad,runname,mode,model,dataset,datasize,batch,T,theta,rhs_mode)
										# os.system("python -W ignore ex_MNIST_CIFAR.py %s"%(options))

								# load accuracy
								with open(Path(dataset+'_results','output','name2accuracy'),'rb') as f: accuracy = pickle.load(f)

								# record accuracy of trained model for various noise levels
								for eps in [0.0,0.1,0.2,0.3]:
									# top-1 accuracy
									acc_key = "train_GN_%.1f-%s"%(eps,runname)
									key = "a_%d_%.2f"%(T,theta)
									acc = accuracy[acc_key][0] if acc_key in accuracy.keys() else np.nan
									acc_vs_eps[key] = acc_vs_eps[key]+[acc] if key in acc_vs_eps.keys() else [acc]
		table = np.stack(list(acc_vs_eps.values()), axis=1)
		header = " ".join(list(acc_vs_eps.keys()))
		np.savetxt( Path(dataset+'_results','output','data','%s_search.txt'%(dataset)), table, delimiter=' ', fmt='%.2f', header=header, comments='' )



if __name__ == '__main__':
	# hyperparameters_search()
	center_search()
	# radius_search()
	# search()