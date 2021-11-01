from pathlib import Path
import numpy as np
import pickle
import re


topk = 1
stds   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] #, 0.6, 0.7, 0.8, 0.9, 1.0]
thetas = [0.0, 0.25, 0.50, 0.75, 1.00]
lims   = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
# lims   = [-1.0]
batches = [100]
datasizes = [1000]
# attacks = ['GN', 'GNclip', 'SP']
attacks = ['GNclip']
times = [1, 3]
adivs = [1]
# lims = [-0.8, -0.4, 0.0]

num_thetas = len(thetas)
num_stds   = len(stds)
# table = {
# 	'train': np.zeros((num_stds,3+num_thetas)),
# 	'valid': np.zeros((num_stds,3+num_thetas)),
# 	}
# best_table = {
# 	'train': np.zeros((num_stds,1)),
# 	'valid': np.zeros((num_stds,1)),
# 	}

with open(Path('output','name2accuracy'),'rb') as f:
	name2accuracy = pickle.load(f)


###############################################################################
###############################################################################
# accuracy vs std and theta


def make_latex_table(num_cols,title,cols_header):
	# thetas_str = ' & '.join(['%0.2f'%(theta) for theta in thetas[1:] ])
	table_header = (
		"\\begin{tabular}{|%s} \n"
		"\\cline{1-%d}"
		"\\multirow{2}{*}{\shortstack[c]{Noise\\\$\epsilon$}} & \\multicolumn{%d}{c|}{%s} \\\ \n\\cline{2-%d}"
		# "& $\\theta=0$  &  %s \\\ \\hline \n"
		" %s \\\ \\hline \n"
		)%((num_cols+1)*'c|',num_cols+1,num_cols,title,num_cols+1,cols_header)
	table_footer = "\\hline\n\\end{tabular}"
	table_body = ""
	for row, val_row in zip(table['train'], table['valid']):
		max_ind     = np.argmax(row)
		max_val_ind = np.argmax(val_row)
		table_row   = "%0.2f"%(row[0])
		for c in range(1,len(row)):
			table_row += " & \\textbf{%0.2f}"%(row[c]) if max_ind==c else " & %0.2f"%(row[c])
			table_row +=  " (\\textbf{%0.2f})"%(val_row[c]) if max_val_ind==c else " (%0.2f)"%(val_row[c])
		table_body += table_row + " \\\ \n"
	return table_header+table_body+table_footer



# ###############################################################################
# # results for plain and 1Lip
# table = {
# 	'train': np.zeros((num_stds,1+num_thetas)),
# 	'valid': np.zeros((num_stds,1+num_thetas)),
# 	}
# # for arch in ['plain', '1Lip']:
# for arch in ['plain']:
# 	for datasize in datasizes:
# 		for batch in batches:
# 			for noise in attacks:
# 				for T in times:
# 					for k in range(topk):
# 						for mode in ['train', 'valid']:
# 							for i, std in enumerate(stds):
# 								table[mode][i,0] = std
# 								for j, theta in enumerate(thetas):
# 									runname = 'clean_data_%d_batch_%d_T_%d_%s_theta_%.2f%s%s%.1f'%(datasize,batch,T,arch,theta,mode,noise,std)
# 									if runname in name2accuracy.keys():
# 										# name = 'clean_data_%d_batch_%d_T_%d_%s_theta_%.2f%s%s%.1f'%(datasize,batch,T,arch,theta,mode,noise,std)
# 										table[mode][i,1+j] = name2accuracy[runname][k]
# 									else:
# 										table[mode][i,1+j] = 0.0
# 							# save table to file
# 							np.savetxt( Path('output','data','%s_top_%d_accuracy_clean_data_%d_batch_%d_T_%d_%s_%s_noise.txt'%(mode,k+1,datasize,batch,T,arch,noise)), table[mode], delimiter=',', fmt='%0.2f' )
# 						# compose latex table
# 						cols_header = '& $\\theta=0$  &' + ' & '.join(['%0.2f'%(theta) for theta in thetas[1:] ])
# 						title = "Top-%d accuracy"%(k+1)
# 						latex_table = make_latex_table(num_thetas,title,cols_header)
# 						# save latex table to file
# 						with open(Path('output','data','table_top_%d_data_%d_batch_%d_T_%d_%s_%s_noise.txt'%(k+1,datasize,batch,T,arch,noise)), "w") as f:
# 							f.write(latex_table)

for arch in ['1Lip']:
	for datasize in datasizes:
		for batch in batches:
			for noise in attacks:
				for T in times:
					for adiv in [0.00]:
						for k in range(topk):
							for mode in ['train', 'valid']:
								for i, std in enumerate(stds):
									table[mode][i,0] = std
									for j, theta in enumerate(thetas):
										runname = 'clean_data_%d_batch_%d_T_%d_adiv_%.2f_theta_%.2f_%s%s%s%.1f'%(datasize,batch,T,adiv,theta,arch,mode,noise,std)
										if runname in name2accuracy.keys():
											# name = 'clean_data_%d_batch_%d_T_%d_%s_theta_%.2f%s%s%.1f'%(datasize,batch,T,arch,theta,mode,noise,std)
											table[mode][i,1+j] = name2accuracy[runname][k]
										else:
											table[mode][i,1+j] = 0.0
								# save table to file
								np.savetxt( Path('output','data','%s_top_%d_accuracy_clean_data_%d_batch_%d_T_%d_%s_%s_noise.txt'%(mode,k+1,datasize,batch,T,arch,noise)), table[mode], delimiter=',', fmt='%0.2f' )
							# compose latex table
							cols_header = '& $\\theta=0$  &' + ' & '.join(['%0.2f'%(theta) for theta in thetas[1:] ])
							title = "Top-%d accuracy"%(k+1)
							latex_table = make_latex_table(num_thetas,title,cols_header)
							# save latex table to file
							with open(Path('output','data','table_top_%d_data_%d_batch_%d_T_%d_%s_%s_noise.txt'%(k+1,datasize,batch,T,arch,noise)), "w") as f:
								f.write(latex_table)

# exit()
# ##############################################################################
# # results for theta scheme with regularization
# table = {
# 	'train': np.zeros((num_stds,1+num_thetas)),
# 	'valid': np.zeros((num_stds,1+num_thetas)),
# 	}
# for datasize in datasizes:
# 	for batch in batches:
# 		for noise in ['GN']:
# 			for T in [1]:
# 				for adiv in [0.00, 1.00]:
# 					for k in range(topk):
# 						for lim in lims if adiv>0 else [-1.0]:
# 							table['train'] *= 0
# 							table['valid'] *= 0
# 							for mode in ['train', 'valid']:
# 								for i, std in enumerate(stds):
# 									table[mode][i,0] = std
# 									# name = 'clean_data_%d_T_%d_plain_theta_0.00%s%s%.1f'%(datasize,T,mode,noise,std)
# 									# table[mode][i,1] = name2accuracy[name][k]
# 									# name = 'clean_data_%d_batch_%d_T_%d_1Lip_theta_0.00%s%s%.1f'%(datasize,batch,T,mode,noise,std)
# 									# table[mode][i,1] = name2accuracy[name][k]
# 									for j, theta in enumerate(thetas):
# 										runname = 'clean_data_%d_batch_%d_T_%d_adiv_%.2f_theta_%.2f_lim_%.1f%s%s%.1f'%(datasize,batch,T,adiv,theta,lim,mode,noise,std)
# 										if runname in name2accuracy.keys():
# 											table[mode][i,1+j] = name2accuracy[runname][k]
# 								# save table to file
# 								np.savetxt( Path('output','data','%s_top_%d_accuracy_clean_data_%d_batch_%d_T_%d_adiv_%0.2f_lim_%.1f_%s_noise.txt'%(mode,k+1,datasize,batch,T,adiv,lim,noise)), table[mode], delimiter=',', fmt='%0.2f' )
# 							# compose latex table
# 							title = "Top-%d accuracy, lim=$%.1f$"%(k+1, lim)
# 							cols_header = ' &  $\\theta=0$  &' + ' & '.join(['%0.2f'%(theta) for theta in thetas[1:] ])
# 							latex_table = make_latex_table(num_thetas+1,title,cols_header)
# 							# save latex table to file
# 							with open(Path('output','data','table_top_%d_data_%d_batch_%d_T_%d_adiv_%0.2f_lim_%.1f_%s_noise.txt'%(k+1,datasize,batch,T,adiv,lim,noise)), "w") as f:
# 								f.write(latex_table)
# # exit()


# ###############################################################################
# # results for theta scheme with regularization
# table = {
# 	'train': np.zeros((num_stds,1+num_thetas)),
# 	'valid': np.zeros((num_stds,1+num_thetas)),
# 	}
# for datasize in datasizes:
# 	for batch in batches:
# 		for noise in attacks:
# 			for T in times:
# 				for adiv in adivs:
# 					for k in range(topk):
# 						table['train'] *= 0
# 						table['valid'] *= 0
# 						for mode in ['train', 'valid']:
# 							for i, std in enumerate(stds):
# 								table[mode][i,0] = std
# 								# name = 'clean_data_%d_T_%d_plain_theta_0.00%s%s%.1f'%(datasize,T,mode,noise,std)
# 								# table[mode][i,1] = name2accuracy[name][k]
# 								# name = 'clean_data_%d_batch_%d_T_%d_1Lip_theta_0.00%s%s%.1f'%(datasize,batch,T,mode,noise,std)
# 								# table[mode][i,1] = name2accuracy[name][k]
# 								for j, theta in enumerate(thetas):
# 									if adiv>0:
# 										runname = 'clean_data_%d_batch_%d_T_%d_adiv_%.2f_theta_%.2f_minstab_0.0%s%s%.1f'%(datasize,batch,T,adiv,theta,mode,noise,std)
# 									else:
# 										runname = 'clean_data_%d_batch_%d_T_%d_adiv_%.2f_theta_%.2f%s%s%.1f'%(datasize,batch,T,adiv,theta,mode,noise,std)
# 									if runname in name2accuracy.keys():
# 										table[mode][i,1+j] = name2accuracy[runname][k]
# 							# save table to file
# 							np.savetxt( Path('output','data','%s_top_%d_accuracy_clean_data_%d_batch_%d_T_%d_adiv_%0.2f_%s_noise.txt'%(mode,k+1,datasize,batch,T,adiv,noise)), table[mode], delimiter=',', fmt='%0.2f' )
# 						# # compose latex table
# 						# title = "Top-%d accuracy, lim=$%.1f$"%(k+1, lim)
# 						# cols_header = ' &  $\\theta=0$  &' + ' & '.join(['%0.2f'%(theta) for theta in thetas[1:] ])
# 						# latex_table = make_latex_table(num_thetas+1,title,cols_header)
# 						# # save latex table to file
# 						# with open(Path('output','data','table_top_%d_data_%d_batch_%d_T_%d_adiv_%0.2f_lim_%.1f_%s_noise.txt'%(k+1,datasize,batch,T,adiv,lim,noise)), "w") as f:
# 						# 	f.write(latex_table)
# exit()

###############################################################################
###############################################################################

# regex for keys to ignore
ignore_keys = '.*batch_10_T_3.*theta_0.75_lim_(-0.4|-0.2).*'

# best accuracy over theta and eigmin/eigmax
best_theta_eigmin = { 'train': np.zeros((num_stds,2)),            'valid': np.zeros((num_stds,2))            }
best_eigmin       = { 'train': np.zeros((num_thetas,num_stds,2)), 'valid': np.zeros((num_thetas,num_stds,2)) }
for datasize in datasizes:
	for batch in batches:
		for T in times:
			for noise in attacks:

				# accuracy as a function of two hyperparameters: lower stability limit and theta
				for adiv in adivs:
					for mode in ['train', 'valid']:
						best_theta_eigmin[mode] *= 0
						for std_i, std in enumerate(stds):
							fname = Path('output','data','%s_accuracy_clean_data_%d_batch_%d_T_%d_adiv_%0.2f_%s_noise_std_%.1f.txt'%(mode,datasize,batch,T,adiv,noise,std))
							with open(fname, 'w') as f:
								f.write("")
							for theta_i, theta in enumerate(thetas):
								# hname = Path('output','data','hist_%s_accuracy_clean_data_%d_batch_%d_T_%d_adiv_%0.2f_theta_%.2f_%s_noise_std_%.1f.txt'%(mode,datasize,batch,T,adiv,theta,noise,std))
								# with open(hname, 'w') as f:
								# 	f.write("")
								best_eigmin[mode] *= 0
								for stabmin in lims:
									runname = 'clean_data_%d_batch_%d_T_%d_adiv_%.2f_theta_%.2f_lim_%.1f%s%s%.1f'%(datasize,batch,T,adiv,theta,stabmin,mode,noise,std)
									if runname in name2accuracy.keys() and not re.search(ignore_keys, runname):
										value = name2accuracy[runname][0]
										best_theta_eigmin[mode][std_i,:]   = std, max(best_theta_eigmin[mode][std_i,1],   value)
										best_eigmin[mode][theta_i,std_i,:] = std, max(best_eigmin[mode][theta_i,std_i,1], value)
										# hist = np.loadtxt(Path('output','data','hist','hist_stab_clean_data_%d_batch_%d_T_%d_adiv_%.2f_theta_%.2f_lim_%.1f_0.ode.txt'%(datasize,batch,T,adiv,theta,stabmin)), delimiter=',' )
										# topbin = np.argmax(hist, 0)[1]
										# topbin = hist[np.argmax(hist, 0)[1],0]
										# accuracy vs histogram
										# with open(hname, 'a') as f: np.savetxt( f, np.array([topbin, value]).reshape((1,-1)), fmt='%.2e', delimiter=' ')
										# accuracy vs stabmin
										with open(fname, 'a') as f: np.savetxt( f, np.array([stabmin, theta, value]).reshape((1,-1)), fmt='%.2e', delimiter=' ')
									else:
										# accuracy vs histogram
										# with open(hname, 'w') as f: f.write("%.2e nan \n"%(topbin))
										# accuracy vs stabmin
										with open(fname, 'a') as f: f.write("%.2e %.2e nan \n"%(stabmin, theta))
								with open(fname, 'a') as f:
									f.write("\n")
						np.savetxt( Path('output','data','best_%s_accuracy_clean_data_%d_batch_%d_T_%d_adiv_%0.2f_%s_noise.txt'%(mode,datasize,batch,T,adiv,noise)), best_theta_eigmin[mode], fmt='%.2e', delimiter=',')
						for theta_i, theta in enumerate(thetas):
							np.savetxt( Path('output','data','best_%s_accuracy_clean_data_%d_batch_%d_T_%d_adiv_%0.2f_theta_%.2f_%s_noise.txt'%(mode,datasize,batch,T,adiv,theta,noise)), best_eigmin[mode][theta_i], fmt='%.2e', delimiter=',')


				# # accuracy as a function of noise std and theta
				# for augm in ['clean']:
				# 	for adiv in [0.0, 1.0]:
				# 		for mode in ['train', 'valid']:
				# 			fname = Path('output','data','%s_accuracy_%s_data_%d_batch_%d_T_%d_adiv_%0.2f_%s_noise.txt'%(mode,augm,datasize,batch,T,adiv,noise))
				# 			with open(fname, 'w') as f:
				# 				f.write("")
				# 			for theta_i, theta in enumerate(thetas):
				# 				for std in stds:
				# 					if adiv>0:
				# 						runname = '%s_data_%d_batch_%d_T_%d_adiv_%.2f_theta_%.2f_minstab_%.1f%s%s%.1f'%(augm,datasize,batch,T,adiv,theta,0.0,mode,noise,std)
				# 					else:
				# 						runname = '%s_data_%d_batch_%d_T_%d_adiv_%.2f_theta_%.2f%s%s%.1f'%(augm,datasize,batch,T,adiv,theta,mode,noise,std)
				# 					if runname in name2accuracy.keys() and not re.search(ignore_keys, runname):
				# 						value = name2accuracy[runname][0]
				# 						with open(fname, 'a') as f:
				# 							np.savetxt( f, np.array([std, theta, value]).reshape((1,-1)), fmt='%.2e', delimiter=' ')
				# 					else:
				# 						with open(fname, 'a') as f:
				# 							f.write("%.2e %.2e nan \n"%(std, theta))
				# 				with open(fname, 'a') as f:
				# 					f.write("\n")


				# # accuracy with 1Lip rhs as a function of noise std and theta
				# for augm in ['clean']:
				# 	for adiv in [1.0]:
				# 		for mode in ['train', 'valid']:
				# 			fname = Path('output','data','%s_accuracy_%s_data_%d_batch_%d_T_%d_adiv_%0.2f_%s_noise_1Lip.txt'%(mode,augm,datasize,batch,T,adiv,noise))
				# 			with open(fname, 'w') as f:
				# 				f.write("")
				# 			for theta_i, theta in enumerate(thetas):
				# 				for std in stds:
				# 					runname = '%s_data_%d_batch_%d_T_%d_adiv_%.2f_theta_%.2f_1Lip%s%s%.1f'%(augm,datasize,batch,T,adiv,theta,mode,noise,std)
				# 					if runname in name2accuracy.keys() and not re.search(ignore_keys, runname):
				# 						value = name2accuracy[runname][0]
				# 						with open(fname, 'a') as f:
				# 							np.savetxt( f, np.array([std, theta, value]).reshape((1,-1)), fmt='%.2e', delimiter=' ')
				# 					else:
				# 						with open(fname, 'a') as f:
				# 							f.write("%.2e %.2e nan \n"%(std, theta))
				# 				with open(fname, 'a') as f:
				# 					f.write("\n")


				# for adiv in adivs:
				# 	for mode in ['train', 'valid']:
				# 		for std_i, std in enumerate(stds):
				# 			fname = Path('output','data','%s_accuracy_clean_data_%d_batch_%d_T_%d_adiv_%0.2f_%s_noise_std_%.1f_minstab.txt'%(mode,datasize,batch,T,adiv,noise,std))
				# 			with open(fname, 'w') as f:
				# 				f.write("")
				# 			for theta_i, theta in enumerate(thetas):
				# 				for stabmin in [2.0]:
				# 					runname = 'clean_data_%d_batch_%d_T_%d_adiv_%.2f_theta_%.2f_minstab_%.1f%s%s%.1f'%(datasize,batch,T,adiv,theta,0.0,mode,noise,std)
				# 					if runname in name2accuracy.keys() and not re.search(ignore_keys, runname):
				# 						value = name2accuracy[runname][0]
				# 						with open(fname, 'a') as f:
				# 							np.savetxt( f, np.array([stabmin, theta, value]).reshape((1,-1)), fmt='%.2e', delimiter=' ')
				# 					else:
				# 						with open(fname, 'a') as f:
				# 							f.write("%.2e %.2e nan \n"%(stabmin, theta))
				# 				with open(fname, 'a') as f:
				# 					f.write("\n")
