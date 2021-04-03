from pathlib import Path
import numpy as np
import pickle


stds   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
thetas = [0.0,0.25,0.50,0.75,1.00]
topk = 2

lims = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]

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
		table_row  = "%0.2f"%(row[0])
		for c in range(1,len(row)):
			table_row += " & \\textbf{%0.2f}"%(row[c]) if max_ind==c else " & %0.2f"%(row[c])
			table_row +=  " (\\textbf{%0.2f})"%(val_row[c]) if max_val_ind==c else " (%0.2f)"%(val_row[c])
		table_body += table_row + " \\\ \n"
	return table_header+table_body+table_footer


###############################################################################
# results for plain and 1Lip
table = {
	'train': np.zeros((num_stds,1+num_thetas)),
	'valid': np.zeros((num_stds,1+num_thetas)),
	}
for arch in ['plain', '1Lip']:
	for datasize in [1000]:
		for noise in ['GN']:
			for T in [1,3]:
				for k in range(topk):
					for mode in ['train', 'valid']:
						for i, std in enumerate(stds):
							table[mode][i,0] = std
							for j, theta in enumerate(thetas):
								name = 'clean_data_1000_T_%d_%s_theta_%.2f%s%s%.1f'%(T,arch,theta,mode,noise,std)
								table[mode][i,1+j] = name2accuracy[name][k]
						# save table to file
						np.savetxt( Path('output','data','%s_top_%d_accuracy_clean_data_%d_T_%d_%s_%s_noise.txt'%(mode,k+1,datasize,T,arch,noise)), table[mode], delimiter=',', fmt='%0.2f' )
					# compose latex table
					cols_header = '& $\\theta=0$  &' + ' & '.join(['%0.2f'%(theta) for theta in thetas[1:] ])
					title = "Top-%d accuracy"%(k+1)
					latex_table = make_latex_table(num_thetas,title,cols_header)
					# save latex table to file
					with open(Path('output','data','table_top_%d_data_%d_T_%d_%s_%s_noise.txt'%(k+1,datasize,T,arch,noise)), "w") as f:
						f.write(latex_table)


###############################################################################
# results for theta scheme with regularization
table = {
	'train': np.zeros((num_stds,3+num_thetas)),
	'valid': np.zeros((num_stds,3+num_thetas)),
	}
for datasize in [1000]:
	for noise in ['GN']:
		for T in [1,3]:
			for adiv in [1.00]:
				for k in range(topk):
					for lim in lims:
						for mode in ['train', 'valid']:
							for i, std in enumerate(stds):
								table[mode][i,0] = std
								name = 'clean_data_1000_T_%d_plain%s%s%.1f'%(T,mode,noise,std)
								table[mode][i,1] = name2accuracy[name][k]
								name = 'clean_data_1000_T_%d_1Lip%s%s%.1f'%(T,mode,noise,std)
								table[mode][i,2] = name2accuracy[name][k]
								for j, theta in enumerate(thetas):
									name = 'clean_data_1000_T_%d_adiv_%.2f_theta_%.2f_lim_%.1f%s%s%.1f'%(T,adiv,theta,lim,mode,noise,std)
									table[mode][i,3+j] = name2accuracy[name][k]
							# save table to file
							np.savetxt( Path('output','data','%s_top_%d_accuracy_clean_data_%d_T_%d_adiv_%0.2f_lim_%.1f_%s_noise.txt'%(mode,k+1,datasize,T,adiv,lim,noise)), table[mode], delimiter=',', fmt='%0.2f' )
						# compose latex table
						title = "Top-%d accuracy, lim=$%.1f$"%(k+1, lim)
						cols_header = '& plain & 1Lip &  $\\theta=0$  &' + ' & '.join(['%0.2f'%(theta) for theta in thetas[1:] ])
						latex_table = make_latex_table(num_thetas+2,title,cols_header)
						# save latex table to file
						with open(Path('output','data','table_top_%d_data_%d_T_%d_adiv_%0.2f_lim_%.1f_%s_noise.txt'%(k+1,datasize,T,adiv,lim,noise)), "w") as f:
							f.write(latex_table)



###############################################################################
###############################################################################
# accuracy as a function of two hyperparameters: lower stability limit and theta

best_table = {
	'train': np.zeros((num_stds,2)),
	'valid': np.zeros((num_stds,2)),
	}
for datasize in [1000]:
	for adiv in [1.00]:
		for T in [1,3]:
			best_table['train'] *= 0
			best_table['valid'] *= 0
			for noise in ['GN']:
				for mode in ['train', 'valid']:
					for std_i, std in enumerate(stds):
						fname = Path('output','data','%s_accuracy_clean_data_%d_T_%d_adiv_%0.2f_%s_noise_std_%.1f.txt'%(mode,datasize,T,adiv,noise,std))
						with open(fname, 'w') as f: f.write("")
						for i, lim in enumerate(lims):
							# prev = 0
							for j, theta in enumerate(thetas):
								name  = 'clean_data_1000_T_%d_adiv_%.2f_theta_%.2f_lim_%.1f%s%s%.1f'%(T,adiv,theta,lim,mode,noise,std)
								value = name2accuracy[name][0]
								best_table[mode][std_i,0] = std
								best_table[mode][std_i,1] = max(best_table[mode][std_i,1], value)
								# if name2accuracy[name][0]>0.5*prev:
									# prev = name2accuracy[name][0]
								with open(fname, 'a') as f: np.savetxt( f, np.array([lim, theta, value]).reshape((1,-1)), fmt='%.2e', delimiter=' ')
							# with open(fname, 'a') as f: np.savetxt( f, np.array([lim, theta, name2accuracy[name][0]]).reshape((1,-1)), fmt='%.2e', delimiter=' ')
							with open(fname, 'a') as f: f.write("\n")
					np.savetxt( Path('output','data','best_%s_accuracy_clean_data_%d_T_%d_adiv_%0.2f_%s_noise.txt'%(mode,datasize,T,adiv,noise)), best_table[mode], fmt='%.2e', delimiter=',')