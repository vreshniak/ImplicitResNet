import warnings
import torch


def rotmat(a,b):
	"""
	Adapted from http://www.netlib.org/templates/matlab/rotmat.m

	Compute the Givens rotation matrix parameters for a and b.
	"""
	if b==0.0:
		c = 1.0
		s = 0.0
	elif b.abs()>a.abs():
		temp = a / b
		s = 1.0 / torch.sqrt(1.0+temp**2)
		c = temp * s
	else:
		temp = b / a
		c = 1.0 / torch.sqrt(1.0+temp**2)
		s = temp * c
	return c, s


def gmres( A, x, b, restrt, max_it, tol, M=None ):
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
	%         restrt   INTEGER number of iterations between restarts
	%         max_it   INTEGER maximum number of iterations
	%         tol      REAL error tolerance
	%
	% output  x        REAL solution vector
	%         error    REAL error norm
	%         iter     INTEGER number of iterations performed
	%         flag     INTEGER: 0 = solution found to tolerance
	%                           1 = no convergence given max_it
	"""

	dtype  = x.dtype
	device = x.device

	# dummy preconditioner
	if M is None: M = lambda x: x

	# initialization
	iter = 0
	flag = 0

	bnrm2 = b.norm()
	if bnrm2==0.0: bnrm2 = 1.0

	# terminate if tolerance achieved
	r = M(b-A(x))
	error = r.norm() / bnrm2
	if error<tol: return x, error, iter, flag


	# initialize workspace
	n, m = x.numel(), restrt
	assert m<=n, "Number of inner iterations should not exceed size of the problem"
	max_it = max_it if m<n else 1

	# Krylov subspace orthogonal basis matrix
	Q  = torch.zeros((n,m+1), dtype=dtype, device=device)
	# Hessenberg matrix
	H  = torch.zeros((m+1,m), dtype=dtype, device=device)
	cs = torch.zeros((m,),    dtype=dtype, device=device)
	sn = torch.zeros((m,),    dtype=dtype, device=device)
	e1 = torch.zeros((n,),   dtype=dtype, device=device)
	e1[0] = 1.0

	# perform outer iterations
	for iter in range(max_it):

		r = M(b-A(x))

		# first basis vector
		Q[:,0] = r / r.norm()

		s = r.norm() * e1

		# restart method and perform inner iterations
		for i in range(m):

			################################################
			# find next basis vector with Arnoldi iteration
			# Krylov vector
			w = M(A(Q[:,i]))
			for k in range(i+1):
				H[k,i] = (w*Q[:,k]).sum()
				w -= H[k,i]*Q[:,k]
			H[i+1,i] = w.norm()
			Q[:,i+1] = w / H[i+1,i]


			################################################
			# apply Givens rotation to eliminate the last element in H ith row
			# rotate kth column
			for k in range(i):
				temp     =  cs[k]*H[k,i] + sn[k]*H[k+1,i]
				H[k+1,i] = -sn[k]*H[k,i] + cs[k]*H[k+1,i]
				H[k,i]   = temp

			# form i-th rotation matrix
			cs[i], sn[i] = rotmat( H[i,i], H[i+1,i] )

			# eliminate H[i+1,i]
			H[i,i]   = cs[i] * H[i,i] + sn[i] * H[i+1,i]
			H[i+1,i] = 0.0

			# update the residual vector
			s[i+1] = -sn[i]*s[i]
			s[i]   =  cs[i]*s[i]
			error  = s[i+1].abs() / bnrm2
			if error <= tol:
				y, _ = torch.solve(s[:i+1].unsqueeze(1), H[:i+1,:i+1])
				x = x + Q[:,:i+1]@y.flatten()
				break

		if error <= tol: break
		y, _ = torch.solve(s[:m].unsqueeze(1), H[:m,:m])
		x = x + Q[:,:m]@y.flatten()
		r = M(b-A(x))
		s[i+1] = r.norm()
		error = s[i+1] / bnrm2
		if error<=tol: break

	if error>tol: flag = 1

	return x, error, iter+1, flag


