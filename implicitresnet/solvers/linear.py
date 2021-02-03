# import warnings
import torch

# from .nonlinear import lbfgs
# import scipy.sparse.linalg as sla
# import numpy as np


def rotmat(a,b):
	"""
	Adapted from http://www.netlib.org/templates/matlab/rotmat.m

	Compute the Givens rotation matrix parameters for a and b.
	"""
	c = torch.zeros_like(a)
	s = torch.zeros_like(a)
	temp = torch.zeros_like(a)

	mask = (b.abs()>a.abs())
	temp[mask] = a[mask] / b[mask]
	s[mask] = 1.0 / torch.sqrt(1.0+temp[mask]**2)
	c[mask] = temp[mask] * s[mask]

	mask = (b.abs()<=a.abs())
	temp[mask] = b[mask] / a[mask]
	c[mask] = 1.0 / torch.sqrt(1.0+temp[mask]**2)
	s[mask] = temp[mask] * c[mask]

	mask = (b==0)
	c[mask] = 1.0
	s[mask] = 0.0

	# if b==0.0:
	# 	c = 1.0
	# 	s = 0.0
	# elif b.abs()>a.abs():
	# 	temp = a / b
	# 	s = 1.0 / torch.sqrt(1.0+temp**2)
	# 	c = temp * s
	# else:
	# 	temp = b / a
	# 	c = 1.0 / torch.sqrt(1.0+temp**2)
	# 	s = temp * c
	return c, s


def gmres( A, x, b, max_iters=None, min_iters=3, max_restarts=1, tol=None, M=None ):
	"""
	Adapted from http://www.netlib.org/templates/matlab/gmres.m

	%  -- Iterative template routine --
	%     Univ. of Tennessee and Oak Ridge National Laboratory
	%     October 1, 1993
	%     Details of this algorithm are described in "Templates for the
	%     Solution of Linear Systems: Building Blocks for Iterative
	%     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
	%     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
	%     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
	%
	% [x, error, iter, flag] = gmres( A, x, b, M, restrt, max_it, tol )
	%
	% gmres.m solves the linear system Ax=b
	% using the Generalized Minimal residual ( GMRESm ) method with restarts .
	%
	% input   A        REAL nonsymmetric positive definite matrix
	%         x        REAL initial guess vector
	%         b        REAL right hand side vector
	%         M        REAL preconditioner matrix
	%         max_iters   INTEGER number of iterations between restarts
	%         max_restarts   INTEGER maximum number of iterations
	%         tol      REAL error tolerance
	%
	% output  x        REAL solution vector
	%         error    REAL error norm
	%         iter     INTEGER number of iterations performed
	%         flag     INTEGER: 0 = solution found to tolerance
	%                           1 = no convergence given max_it
	"""

	# dummy preconditioner (might replace with something real later)
	if M is None: M = lambda x: x

	assert x.ndim==2, "x must have batch dimension, x.ndim = "+str(x.ndim)
	assert b.ndim==2, "b must have batch dimension, b.ndim = "+str(b.ndim)

	# dimensions, dtype and device of the problem
	batch_dim = x.size(0)
	n         = x.size(1)
	dtype     = x.dtype
	device    = x.device

	if tol is None: tol = 1*torch.finfo(dtype).eps
	tol = max(tol*b.norm(1).amax(), tol)

	# set max_iters if not given, and perform sanity checks
	assert max_restarts>0,  "max_restarts must be greater than 0, max_restarts = "+str(max_restarts)
	assert max_restarts<=n, "max_restarts should not exceed size of the problem n, max_restarts = "+str(max_restarts)
	if max_iters is None: max_iters = n//max_restarts
	if max_iters<n:
		max_restarts = n//max_iters + 1
	elif max_iters>=n:
		max_iters    = n
		max_restarts = 1

	# initialization
	iters = 0
	flag  = 0

	# norm of the RHS
	bnrm2 = b.norm(dim=1)
	bnrm2[bnrm2==0.0] = 1.0

	# terminate if tolerance achieved
	# r = M(b-A(x))
	# error = r.norm(dim=1) / bnrm2
	# error = r.norm(dim=1)
	# if error.amax()<tol: return x, error.amax(), iters, flag


	# initialize workspace
	# orthogonal basis matrix of the Krylov subspace
	Q  = torch.zeros((batch_dim,n,max_iters+1), dtype=dtype, device=device)
	# H is upper Hessenberg matrix, H is A on basis Q
	H  = torch.zeros((batch_dim,max_iters+1,max_iters), dtype=dtype, device=device)
	# cosines and sines of the rotation matrix
	cs = torch.zeros((batch_dim,max_iters,), dtype=dtype, device=device)
	sn = torch.zeros((batch_dim,max_iters,), dtype=dtype, device=device)
	#
	e1 = torch.zeros((batch_dim,n+1,), dtype=dtype, device=device)
	e1[:,0] = 1.0

	# perform outer iterations
	for _ in range(max_restarts):

		r = M(b-A(x))
		rnrm2 = r.norm(dim=1,keepdim=True)
		rnrm2[rnrm2==0.0] = 1.0

		# first basis vector
		Q[...,0] = r / rnrm2

		s = rnrm2 * e1

		# restart method and perform inner iterations
		for i in range(max_iters):
			iters += 1

			################################################
			# find next basis vector with Arnoldi iteration

			# (i+1)-st Krylov vector
			w = M(A(Q[...,i]))
			# Gram-Schmidt othogonalization
			for k in range(i+1):
				H[:,k,i] = (w*Q[...,k]).sum(dim=1)
				w -= H[:,k,i].unsqueeze(1) * Q[...,k]
			w += 1.e-12 # to make possible 0/0=1 (Why can this happen?)
			H[:,i+1,i] = w.norm(dim=1)
			# (i+1)-st basis vector
			Q[:,:,i+1] = w / H[:,i+1,i].unsqueeze(1)


			################################################
			# apply Givens rotation to eliminate the last element in H ith row

			# rotate kth column
			for k in range(i):
				temp       =  cs[:,k]*H[:,k,i] + sn[:,k]*H[:,k+1,i]
				H[:,k+1,i] = -sn[:,k]*H[:,k,i] + cs[:,k]*H[:,k+1,i]
				H[:,k,i]   = temp

			# form i-th rotation matrix
			cs[:,i], sn[:,i] = rotmat( H[:,i,i], H[:,i+1,i] )

			# eliminate H[i+1,i]
			H[:,i,i]   = cs[:,i]*H[:,i,i] + sn[:,i]*H[:,i+1,i]
			H[:,i+1,i] = 0.0


			################################################
			# update the residual vector
			s[:,i+1] = -sn[:,i]*s[:,i]
			s[:,i]   =  cs[:,i]*s[:,i]
			error    = s[:,i+1].abs().amax()
			# yy, _ = torch.triangular_solve(s[:,:i+1].unsqueeze(2), H[:,:i+1,:i+1], upper=True)
			# xx = torch.baddbmm(x.unsqueeze(2), Q[:,:,:i+1], yy).squeeze(2)
			# error    = (s[:,i+1].abs()/bnrm2).amax()
			# print(i, "%.2e"%(error.item()), "%.2e"%((M(b-A(xx)).norm(dim=1)).amax().item()))
			if error<tol and (i+1)>min_iters: break

		# update approximation
		y, _ = torch.triangular_solve(s[:,:i+1].unsqueeze(2), H[:,:i+1,:i+1], upper=True)
		x = torch.baddbmm(x.unsqueeze(2), Q[:,:,:i+1], y).squeeze(2)

		r = M(b-A(x))
		error = r.norm(dim=1).amax()
		# s[:,i+1] = r.norm(dim=1)
		# error = (s[:,i+1].abs() / bnrm2).amax()

		if error<tol: break

	if error>tol: flag = 1

	return x, error, iters, flag


def scipy_lgmres( A, x, b, max_iters=None, tol=None, M=None ):
	assert x.ndim==2, "x must have batch dimension, x.ndim = "+str(x.ndim)
	assert b.ndim==2, "b must have batch dimension, b.ndim = "+str(b.ndim)

	# dimensions, dtype and device of the problem
	batch_dim = x.size(0)
	n         = x.size(1)
	ndof      = x.nelement()
	dtype     = x.dtype
	device    = x.device

	iters = [0]
	flag  = 0

	if max_iters is None: max_iters = n
	if max_iters>n:
		max_iters = n

	if tol is None: tol = 10*torch.finfo(dtype).eps

	# torch to numpy dtypes
	numpy_dtype = {torch.float: np.float, torch.float32: np.float32, torch.float64: np.float64}

	# 'Matrix-vector' product of the linear operator
	def matvec(v):
		iters[0] += 1
		v0 = torch.from_numpy(v).view_as(x).to(device=x.device, dtype=x.dtype)
		return A(v0).cpu().detach().numpy().ravel()
	A_dot = sla.LinearOperator(dtype=numpy_dtype[b.dtype], shape=(ndof,ndof), matvec=matvec)

	# Note that norm(residual) <= max(tol*norm(b), atol. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html
	x, info = sla.lgmres( A_dot, b.cpu().detach().numpy().ravel(), x0=b.cpu().detach().numpy().ravel(), maxiter=max_iters, tol=tol, M=None )
	x = torch.from_numpy(x).view_as(b).to(device=b.device, dtype=b.dtype)

	error = (A(x)-b).reshape((b.size(0),-1)).norm(dim=1).max().detach() # \| A*x - b \|
	if error>tol*b.norm(1).amax(): flag=1

	return x, error, iters[0], flag


def linsolve(A, x0, b, method='gmres', **kwargs):
	if method=='gmres':
		return gmres( A, x0, b, **kwargs )
	elif method=='scipy_lgmres':
		return scipy_lgmres( A, x0, b, **kwargs )
	elif method=='lbfgs':
		batch_dim = x0.size(1)
		return lbfgs( lambda x: (A(x)-b).pow(2).reshape((batch_dim,-1)).sum(dim=1), x0, **kwargs )





# class neumann_backprop(Function):
# 	@staticmethod
# 	def forward(ctx, y, y_fp):
# 		# ctx.obj = obj
# 		ctx.save_for_backward(y, y_fp)
# 		return y

# 	@staticmethod
# 	def backward(ctx, dy):
# 		y, y_fp, = ctx.saved_tensors

# 		# residual = lambda dx: (dx-A_dot(dx)-dy).flatten().norm() # \| (I-A) * dx - dy \|
# 		A_dot    = lambda x: torch.autograd.grad(y_fp, y, grad_outputs=x, retain_graph=True, only_inputs=True)[0]
# 		residual = lambda Adx: (Adx-dy).reshape((dy.size()[0],-1)).norm(dim=1).max() #.flatten().norm() # \| (I-A) * dx - dy \|

# 		tol = atol = torch.tensor(_TOL)
# 		TOL = torch.max(tol*dy.norm(), atol)

# 		#######################################################################
# 		# Neumann series

# 		dx  = dy
# 		Ady = A_dot(dy)
# 		Adx = Ady
# 		r1  = residual(dx-Adx)
# 		neu_iters = 1
# 		while r1>=TOL and neu_iters<_max_iters:
# 			r0  = r1
# 			dx  = dx + Ady
# 			Ady = A_dot(Ady)
# 			Adx = Adx + Ady
# 			r1  = residual(dx-Adx)
# 			neu_iters += 1
# 			assert r1<r0, "Neumann series hasn't converged at iteration "+str(neu_iters)+" out of "+str(_max_iters)+" max iterations"

# 		if _collect_stat:
# 			global _backward_stat
# 			_backward_stat['steps']        = _backward_stat.get('steps',0) + 1
# 			_backward_stat['neu_residual'] = _backward_stat.get('neu_residual',0) + r1
# 			_backward_stat['neu_iters']    = _backward_stat.get('neu_iters',0) + neu_iters
# 		return None, dx
