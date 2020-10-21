import numpy as np
import matplotlib.pyplot as plt


class nf(float):
    def __repr__(self):
        s = f'{self:.2f}'
        return f'{self:.0f}' if s[-1]+s[-2] == '0' else s


FeedForward   = lambda z: z
ForwardEuler  = lambda z: 1+z
BackwardEuler = lambda z: 1./(1-z)
Trapezoidal   = lambda z: (2+z)/(2-z)


xlim = (-5,5)
ylim = (-5,5)
X,Y = np.meshgrid(np.linspace(xlim[0],xlim[1],200),np.linspace(ylim[0],ylim[1],200))

def plot_stab(func, fname, fig_no=1, levels=20):
	def no_zero(x):
		if x==0:
			return 1.e-6
		else:
			return x
	Z = abs(func(X+1j*Y))
	z_levels = np.unique(np.hstack((np.linspace(xlim[0],0,5)+1j*0,np.linspace(0,xlim[1],5)+1j*0)))
	zz = abs(func(z_levels))
	levels = np.logspace(np.log10(no_zero(np.amin(zz))), np.log10(no_zero(np.amax(zz))), levels)
	levels = np.unique(np.sort(levels))

	plt.figure(fig_no)
	plt.axhline(y=0, color='black')
	plt.axvline(x=0, color='black')
	ax = plt.gca()
	ax.set_aspect('equal', adjustable='box')
	plt.contourf(X,Y,Z, levels=[0,1], colors='0.6', linewidths=1)
	plt.contour(X,Y,Z, levels=[1], colors='black', linewidths=1)
	cs = plt.contour(X,Y,Z, levels=levels, colors='black')
	cs.levels = [nf(val) for val in cs.levels]
	ax.clabel(cs, cs.levels, inline=True, fmt=r'%r', fontsize=10)
	plt.hlines(0,xlim[0],xlim[1])
	plt.vlines(0,ylim[0],ylim[1])
	plt.show()
	# plt.savefig(fname, bbox_inches='tight')

plot_stab(FeedForward,   'stab_feed_forward.eps', 1, 20)
plot_stab(ForwardEuler,  'stab_resnet.eps',       2, 10)
plot_stab(BackwardEuler, 'stab_backward.eps',     3, 10)
plot_stab(Trapezoidal,   'stab_trapezoidal.eps',  4, 15)

# exit()
# plt.contourf(X,Y,abs(ForwardEuler(Z)), levels=100, linewidths=0.5, cmap="RdBu_r")
# plt.figure(0); plt.contour(X,Y,abs(FeedForward(Z)),   levels=levels)