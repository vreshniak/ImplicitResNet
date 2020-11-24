import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np
import ex_setup

from tensorboard.backend.event_processing import event_accumulator


# https://stackoverflow.com/questions/52756152/tensorboard-extract-scalar-by-a-script
def _load_run(path):
	event_acc = event_accumulator.EventAccumulator(path)
	event_acc.Reload()
	data = {}

	for tag in sorted(event_acc.Tags()["scalars"]):
		# print(tag)
		x, y = [], []

		for scalar_event in event_acc.Scalars(tag):
			x.append(scalar_event.step)
			y.append(scalar_event.value)

		data[tag] = np.hstack( (np.asarray(x).reshape(-1,1), np.asarray(y).reshape(-1,1)) )
	return data


scalars = { '_div': 'rhs/0.ode_div',
			'_jac': 'rhs/0.ode_jac',
			'_loss': 'loss/0_train',
			'_iters': 'stat_forward_train/0.ode_iters',
			'_resid': 'stat_forward_train/0.ode_residual'
		}


logdir  = "./logs"
savedir = "./output/data/"
Path(savedir).mkdir(parents=True, exist_ok=True)


num_points = 50
for run in os.listdir(logdir):
	rundir = os.path.join(logdir, run)
	if os.path.isdir(rundir):
		data = _load_run(rundir)
		args = ex_setup.get_options_from_name(run)
		dataname = "theta%s_T%d_data%d_adiv%s"%( str(args['theta']).replace('.',''), args['T'], args['datasize'], str(args['adiv']).replace('.','') )
		for key, val in scalars.items():
			ind = np.linspace( 0, data[val].shape[0]-1, num_points ).astype('int')
			np.savetxt(savedir+dataname+key+".csv", data[val][ind,:], delimiter=',', header='Step, Value', comments='')


for adiv in [0.0,0.25,0.5,0.75,1.0]:
	iters_vs_theta = []
	for theta in [0.0,0.25,0.5,0.75,1.0]:
		dataname = "theta%s_T5_data%d_adiv%s"%( str(theta).replace('.',''), args['datasize'], str(adiv).replace('.','') )
		data = np.loadtxt(savedir+dataname+"_iters.csv", delimiter=',', skiprows=1)
		iters_vs_theta = iters_vs_theta + [theta, data[-1,1]]
	np.savetxt(savedir+"iters_vs_theta_adiv%s.csv"%(str(adiv).replace('.','')), np.array(iters_vs_theta).reshape((-1,2)), delimiter=',', header='Step, Value', comments='')

for theta in [0.0,0.25,0.5,0.75,1.0]:
	iters_vs_adiv = []
	for adiv in [0.0,0.25,0.5,0.75,1.0]:
		dataname = "theta%s_T5_data%d_adiv%s"%( str(theta).replace('.',''), args['datasize'], str(adiv).replace('.','') )
		data = np.loadtxt(savedir+dataname+"_iters.csv", delimiter=',', skiprows=1)
		iters_vs_adiv = iters_vs_adiv + [adiv, data[-1,1]]
	np.savetxt(savedir+"iters_vs_adiv_theta%s.csv"%(str(theta).replace('.','')), np.array(iters_vs_adiv).reshape((-1,2)), delimiter=',', header='Step, Value', comments='')
