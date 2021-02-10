import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta, abstractmethod

import torch
from implicitresnet.solvers.linear import linsolve


###############################################################################
###############################################################################


_cpu = torch.device('cpu')
_gpu = torch.device('cuda')
_batches = 1


###############################################################################
###############################################################################


def get_system(name, n, dtype, device):
	if name=='wilkinson':
		# https://www.mathworks.com/help/matlab/ref/wilkinson.html
		A = torch.zeros((_batches,n,n)).to(dtype=dtype, device=device)
		for b in range(_batches):
			torch.diagonal(A[b,...], offset=0, dim1=0, dim2=1)[...] = torch.linspace(-(n-1)/2,(n-1)/2,n).to(dtype=dtype, device=device).abs()[...]
		torch.diagonal(A, offset=-1, dim1=1, dim2=2)[...] = torch.ones_like(torch.diagonal(A, offset=1, dim1=1, dim2=2))[...]
		torch.diagonal(A, offset=1,  dim1=1, dim2=2)[...] = torch.ones_like(torch.diagonal(A, offset=1, dim1=1, dim2=2))[...]
		x  = torch.ones((_batches,n)).to(dtype=dtype, device=device)
		# x0 = x + 0.1*torch.randn_like(x)
		x0 = torch.rand((_batches,n)).to(dtype=dtype, device=device)
		# x0 = torch.zeros((_batches,n)).to(dtype=dtype, device=device
		b  = torch.bmm(A,x.unsqueeze(2)).squeeze(2)
		return lambda x:torch.bmm(A,x.unsqueeze(2)).squeeze(2), x0, b
	if name=='laplace':
		pi = 2*3.141592653589793238462643383279502884197169399375105820974944592307816406286
		h = 2*pi/(n+1)
		t = torch.linspace(h,2*pi-h,n)
		A = torch.zeros((_batches,n,n)).to(dtype=dtype, device=device)
		for b in range(_batches):
			torch.diagonal(A[b,...], offset=0, dim1=0, dim2=1)[...] = -2/h**2
		torch.diagonal(A, offset=-1, dim1=1, dim2=2)[...] = 1/h**2
		torch.diagonal(A, offset=1,  dim1=1, dim2=2)[...] = 1/h**2
		x0 = torch.rand((_batches,n)).to(dtype=dtype, device=device)
		# x0 = torch.zeros((_batches,n)).to(dtype=dtype, device=device)
		x = torch.zeros_like(x0)
		for i in range(_batches):
			x[i,...] = torch.sin(t)[...]
		b = torch.bmm(A,x.unsqueeze(2)).squeeze(2)
		return lambda x:torch.bmm(A,x.unsqueeze(2)).squeeze(2), x0, b
	elif name=='random':
		A  = torch.rand((_batches,n,n)).to(dtype=dtype, device=device)
		x0 = torch.rand((_batches,n)).to(dtype=dtype, device=device)
		# x0 = torch.zeros((_batches,n)).to(dtype=dtype, device=device)
		x  = torch.ones((_batches,n)).to(dtype=dtype, device=device)
		b  = torch.bmm(A,x.unsqueeze(2)).squeeze(2)
		return lambda x:torch.bmm(A,x.unsqueeze(2)).squeeze(2), x0, b
	elif name=='mlp_jacobian':
		nn = torch.nn.Sequential(
			torch.nn.Linear(n,2*n,bias=False),
			torch.nn.ReLU(),
			torch.nn.Linear(2*n,2*n,bias=False),
			torch.nn.ReLU(),
			torch.nn.Linear(2*n,n,bias=False)).to(dtype=dtype, device=device)
		input = torch.ones((_batches,n), dtype=dtype, device=device).requires_grad_(True)
		output = nn(input)
		A_dot = lambda x: torch.autograd.grad(output, input, grad_outputs=x, create_graph=True, retain_graph=True, only_inputs=True)[0]
		x0 = torch.rand((_batches,n)).to(dtype=dtype, device=device)
		# x0 = torch.zeros((_batches,n)).to(dtype=dtype, device=device)
		x  = torch.ones((_batches,n)).to(dtype=dtype, device=device)
		b  = A_dot(x)
		return A_dot, x0, b


###############################################################################
###############################################################################


class LinearSolverTest(metaclass=ABCMeta):
	@abstractmethod
	def solve(self, n, dtype, device):
		pass

	@property
	def systems(self):
		return ['wilkinson','random','mlp_jacobian'] #'laplace'

	def test_cpu_n10_float32(self):
		self.solve(10, torch.float32, _cpu)

	def test_cpu_n10_float64(self):
		self.solve(10, torch.float64, _cpu)

	def test_gpu_n10_float32(self):
		if torch.cuda.is_available():
			self.solve(10, torch.float32, _gpu)

	def test_gpu_n10_float64(self):
		if torch.cuda.is_available():
			self.solve(10, torch.float64, _gpu)

	def test_cpu_n100_float32(self):
		self.solve(100, torch.float32, _cpu)

	def test_cpu_n100_float64(self):
		self.solve(100, torch.float64, _cpu)

	def test_gpu_n100_float32(self):
		if torch.cuda.is_available():
			self.solve(100, torch.float32, _gpu)

	def test_gpu_n100_float64(self):
		if torch.cuda.is_available():
			self.solve(100, torch.float64, _gpu)


###############################################################################
###############################################################################


def check_convegence(method, system, error, iters, max_iters, flag):
	if flag==0: print("%s for %12s system converged after %4d/%4d iterations with error %6.3e"%(method, system, iters, max_iters, error))
	assert flag==0, "%s for %12s system failed to converge after %d iterations with error %6.3e"%(method, system, iters, error)

class Test_GMRES_Solver(LinearSolverTest):
	def solve(self, n, dtype, device):
		for system in self.systems:
			A, x0, b = get_system(system, n, dtype, device)
			sol, error, iters, flag = linsolve( A, x0, b, method='gmres', tol=1.e-6)
			check_convegence('GMRES', system, error, iters, n, flag)


# ###############################################################################
# ###############################################################################


# class Test_scipy_lgmres_Solver(LinearSolverTest):
# 	def solve(self, n, dtype, device):
# 		for system in self.systems:
# 			A, x0, b = get_system(system, n, dtype, device)
# 			sol, error, iters, flag = linsolve( A, x0, b, method='scipy_lgmres')
# 			self.check_convegence('scipy_lgmres', system, error, iters, n, flag)


# ###############################################################################
# ###############################################################################


# class Test_LBFGS_Solver(LinearSolverTest):
# 	def solve(self, n, dtype, device):
# 		for system in self.systems:
# 			A, x0, b = get_system(system, n, dtype, device)
# 			sol, error, iters, flag = linsolve( A, x0, b, method='lbfgs', max_iters=1000)
# 			self.check_convegence('LBFGS', system, error, iters, n, flag)