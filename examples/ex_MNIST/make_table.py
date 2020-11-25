from pathlib import Path
import numpy as np


stds   = [0.0,0.1,0.2,0.3,0.4,0.5]
thetas = [0.0,0.25,0.5,0.75,1.0]
topk = 2

table = np.zeros((len(stds),1+len(thetas)*topk))


for i,std in enumerate(stds):
	table[i,0] = std
	for j,theta in enumerate(thetas):
		data = np.loadtxt(Path('output','data',('acc_std%3.1f_theta%4.2f'%(std,theta)).replace('.','')+'.txt'), delimiter=',')
		for k in range(topk):
			table[i,1+k*len(thetas)+j] = data[k]

np.savetxt( Path('output','table.txt'), table, delimiter=' & ', fmt='%4.2f', newline=' \\\\ \n')