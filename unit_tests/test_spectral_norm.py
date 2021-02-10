import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from abc import ABCMeta, abstractmethod

import numpy as np
from pytest import approx

import torch
from implicitresnet.utils.spectral import spectral_norm
import implicitresnet.utils.calc as utils


###############################################################################
###############################################################################


_cpu = torch.device('cpu')
_gpu = torch.device('cuda')
_batches = 1


###############################################################################
###############################################################################


class Test_SpectralNorm:
	def test_matrix_100x100(self):
		A = spectral_norm(torch.nn.Linear(100, 100, bias=False), name='weight', input_shape=(100,), n_power_iterations=1, eps=1e-12, dim=None)
		y = torch.ones(100)
		for _ in range(1000):
			y = A(y)
		y.requires_grad_(True)
		jacobian = utils.jacobian( A(y), y, True ).reshape(y.numel(),y.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		print("spectral norm = %.2e"%(np.amax(singvals)))
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)

	def test_matrix_200x100(self):
		A = spectral_norm(torch.nn.Linear(100, 200, bias=False), name='weight', input_shape=(100,), n_power_iterations=1, eps=1e-12, dim=None)
		x = torch.ones(100)
		for _ in range(1000):
			y = A(x)
		x.requires_grad_(True)
		jacobian = utils.jacobian( A(x), x, True ).reshape(y.numel(),x.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		print("spectral norm = %.2e"%(np.amax(singvals)))
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)

	def test_matrix_100x200(self):
		A = spectral_norm(torch.nn.Linear(200, 100, bias=False), name='weight', input_shape=(200,), n_power_iterations=1, eps=1e-12, dim=None)
		x = torch.ones(200)
		for _ in range(1000):
			y = A(x)
		x.requires_grad_(True)
		jacobian = utils.jacobian( A(x), x, True ).reshape(y.numel(),x.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		print("spectral norm = %.2e"%(np.amax(singvals)))
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)

	def test_conv2d_5_5_28_28(self):
		input_shape = (5,28,28)
		A = spectral_norm(torch.nn.Conv2d(5, 5, kernel_size=3, padding=3//2, bias=False), name='weight', input_shape=input_shape, n_power_iterations=1, eps=1e-12, dim=None)
		x = torch.ones(1,*input_shape)
		for _ in range(1000):
			y = A(x)
		x.requires_grad_(True)
		jacobian = utils.jacobian( A(x), x, True ).reshape(y.numel(),x.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		print("spectral norm = %.3e"%(np.amax(singvals)))
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)

	def test_conv2d_5_3_28_28(self):
		input_shape = (5,28,28)
		A = spectral_norm(torch.nn.Conv2d(5, 3, kernel_size=3, padding=0, bias=False), name='weight', input_shape=input_shape, n_power_iterations=1, eps=1e-12, dim=None)
		x = torch.ones(1,*input_shape)
		for _ in range(1000):
			y = A(x)
		x.requires_grad_(True)
		jacobian = utils.jacobian( A(x), x, True ).reshape(y.numel(),x.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		print("spectral norm = %.3e"%(np.amax(singvals)))
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)

	def test_save_load_state_dict(self):
		A = spectral_norm(torch.nn.Linear(100, 200, bias=False), name='weight', input_shape=(100,), n_power_iterations=1, eps=1e-12, dim=None)
		x = torch.ones(100)
		for _ in range(10):
			y = A(x)
		B = spectral_norm(torch.nn.Linear(100, 200, bias=False), name='weight', input_shape=(100,), n_power_iterations=1, eps=1e-12, dim=None)
		B.load_state_dict(A.state_dict())


# a = Test_SpectralNorm()
# a.save_load_state_dict()