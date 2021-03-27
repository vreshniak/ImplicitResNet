import warnings
import torch


###############################################################################
###############################################################################
# default values

_min_iters    = 3
_max_iters    = 1000
_history_size = 30


###############################################################################
###############################################################################


def lbfgs( fun, x0, tol=None, max_iters=_max_iters, min_iters=_min_iters, history_size=_history_size, batch_error='max' ):
	iters = [0]
	error = [0]
	flag  = 0

	if batch_error=='max':
		batch_err = lambda z: z.amax()
	elif batch_error=='mean':
		batch_err = lambda z: z.mean()

	dtype  = x0.dtype
	device = x0.device

	if tol is None: tol = 10*torch.finfo(dtype).eps

	# check initial residual
	with torch.no_grad():
		error[0] = batch_err(fun(x0))
		if error[0]<tol: return x0.detach(), error[0].detach(), iters[0], flag

	# initial condition: make new (that's why clone) leaf (that's why detach) node which requires gradient
	x = x0.clone().detach().requires_grad_(True)

	nsolver = torch.optim.LBFGS([x], lr=1, max_iter=max_iters, max_eval=None, tolerance_grad=1.e-12, tolerance_change=1.e-12, history_size=history_size, line_search_fn='strong_wolfe')
	def closure():
		resid = fun(x)
		error[0] = batch_err(resid)
		residual = resid.mean()
		nsolver.zero_grad()
		# if error[0]>tol: residual.backward()
		# use .grad() instead of .backward() to avoid evaluation of gradients for leaf parameters which must be frozen inside nsolver
		if error[0]>tol or iters[0]<min_iters: x.grad, = torch.autograd.grad(residual, x, only_inputs=True, allow_unused=False)
		iters[0] += 1
		return residual
	nsolver.step(closure)

	if error[0]>tol: flag=1

	return x.detach(), error[0].detach(), iters[0], flag



def nsolve(fun, x0, method='lbfgs', **kwargs):
	if method=='lbfgs':
		return lbfgs( fun, x0, **kwargs )
