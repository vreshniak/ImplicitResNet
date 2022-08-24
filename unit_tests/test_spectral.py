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
		A = spectral_norm(torch.nn.Linear(100, 100, bias=False), name='weight', n_power_iterations=1)
		y = torch.ones(1,100)
		for _ in range(1000):
			y = A(y)
		y.requires_grad_(True)
		jacobian = utils.jacobian( A(y), y, True ).reshape(y.numel(),y.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)

	def test_matrix_200x100(self):
		A = spectral_norm(torch.nn.Linear(100, 200, bias=False), name='weight', n_power_iterations=1)
		x = torch.ones(1,100)
		for _ in range(1000):
			y = A(x)
		x.requires_grad_(True)
		jacobian = utils.jacobian( A(x), x, True ).reshape(y.numel(),x.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		print("spectral norm = %.2e"%(np.amax(singvals)))
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)

	def test_matrix_100x200(self):
		A = spectral_norm(torch.nn.Linear(200, 100, bias=False), name='weight', n_power_iterations=1)
		x = torch.ones(1,200)
		for _ in range(1000):
			y = A(x)
		x.requires_grad_(True)
		jacobian = utils.jacobian( A(x), x, True ).reshape(y.numel(),x.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		print("spectral norm = %.2e"%(np.amax(singvals)))
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)

	def test_matrix_100x100_sigma_5(self):
		A = spectral_norm(torch.nn.Linear(100, 100, bias=False), name='weight', n_power_iterations=1, sigma=5)
		y = torch.ones(1,100)
		for _ in range(1000):
			y = A(y)
		y.requires_grad_(True)
		jacobian = utils.jacobian( A(y), y, True ).reshape(y.numel(),y.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		assert np.amax(singvals) == approx(5.0, abs=1.e-3)

	def test_conv2d_5_5_28_28(self):
		A = spectral_norm(torch.nn.Conv2d(5, 5, kernel_size=3, padding=3//2, bias=False), name='weight', n_power_iterations=1)
		x = torch.ones(1,5,28,28)
		for _ in range(1000):
			y = A(x)
		x.requires_grad_(True)
		jacobian = utils.jacobian( A(x), x, True ).reshape(y.numel(),x.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		print("spectral norm = %.3e"%(np.amax(singvals)))
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)

	def test_conv2d_5_3_28_28(self):
		A = spectral_norm(torch.nn.Conv2d(5, 3, kernel_size=3, padding=0, bias=False), name='weight', n_power_iterations=1)
		x = torch.ones(1,5,28,28)
		for _ in range(1000):
			y = A(x)
		x.requires_grad_(True)
		jacobian = utils.jacobian( A(x), x, True ).reshape(y.numel(),x.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		print("spectral norm = %.3e"%(np.amax(singvals)))
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)

	def test_save_load_state_dict(self):
		A = spectral_norm(torch.nn.Linear(100, 200, bias=False), name='weight', n_power_iterations=1)
		x = torch.ones(1,100)
		for _ in range(10):
			y = A(x)
		B = spectral_norm(torch.nn.Linear(100, 200, bias=False), name='weight', n_power_iterations=1)
		B.load_state_dict(A.state_dict())
		for _ in range(100):
			y = B(x)
		x.requires_grad_(True)
		jacobian = utils.jacobian( B(x), x, True ).reshape(y.numel(),x.numel()).cpu().detach().numpy()
		singvals = np.linalg.svd(jacobian, compute_uv=False)
		print("spectral norm = %.2e"%(np.amax(singvals)))
		assert np.amax(singvals) == approx(1.0, abs=1.e-3)


# a = Test_SpectralNorm()
# a.save_load_state_dict()