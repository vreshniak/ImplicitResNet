"""
This is extendion of the spectral normalization based on
	https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/spectral_norm.py
Changes:
	1. power iterations are performed on W^t @ W
	2. hence no need for u, only v
	3. convolutional operators normalized correctly rather than normalize 2d matrix representation of the weight tensor
	4. W^t @ W computed as torch.grad of 0.5*||Wx||^2 w.r.t. x
"""



import torch
# from torch.nn.functional import normalize
from typing import Any, Optional, Union, TypeVar
from torch.nn import Module

from .calc import jacobian, hessian

import torch.nn.functional as F


class SpectralNorm:
	name: str
	n_power_iterations: int
	eps: float

	def __init__(self, name: str = 'weight', n_power_iterations: int = 1, eps: float = 1e-12) -> None:
		if n_power_iterations <= 0:
			raise ValueError(f'Expected n_power_iterations to be positive, but got n_power_iterations={n_power_iterations}')
		self.name = name
		self.n_power_iterations = n_power_iterations
		self.eps = eps

	def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
		# NB: If `do_power_iteration` is set, the `v` vector is
		#     updated in power iteration **in-place**. This is very important
		#     because in `DataParallel` forward, the buffers are
		#     broadcast from the parallelized module to each module replica,
		#     which is a new module object created on the fly. And each replica
		#     runs its own spectral norm power iteration. So simply assigning
		#     the updated buffers to the module this function runs on will cause
		#     the update to be lost forever. And the next time the parallelized
		#     module is replicated, the same randomly initialized buffers are
		#     broadcast and used!
		#
		#     Therefore, to make the change propagate back, we rely on two
		#     important behaviors (also enforced via tests):
		#       1. `DataParallel` doesn't clone storage if the broadcast tensor
		#          is already on correct device; and it makes sure that the
		#          parallelized module is already on `device[0]`.
		#       2. If the out tensor in `out=` kwarg has correct shape, it will
		#          just fill in the values.
		#     Therefore, since the same power iteration is performed on all
		#     devices, simply updating the tensors in-place will make sure that
		#     the module replica on `device[0]` will update the _v vector on the
		#     parallized module (by shared storage).
		#
		#    However, after we update `v` in-place, we need to **clone**
		#    it before using it to normalize the weight. This is to support
		#    backproping through two forward passes, e.g., the common pattern in
		#    GAN training: loss = D(real) - D(fake). Otherwise, engine will
		#    complain that variables needed to do backward for the first forward
		#    (i.e., the `v` vector) are changed in the second forward.

		weight      = getattr(module, self.name + '_orig')
		v           = getattr(module, self.name + '_v')
		sigma_range = getattr(module, self.name + '_sigma_range')
		sigma_var   = getattr(module, self.name + '_sigma_var')

		# compute singular vector
		if do_power_iteration:
			with torch.no_grad():
				for _ in range(self.n_power_iterations):
					# Spectral norm of weight equals to `||W v||`, where `v` is the largest singular vector.
					# This power iteration produces approximation of `v` as (W^t W v) / ||W^t W v||.
					with torch.enable_grad():
						v.requires_grad_(True)
						Wv = self.weight_fn(module, weight, v)
						Wv = 0.5 * torch.autograd.grad(Wv.pow(2).sum(), v, create_graph=False, retain_graph=False, only_inputs=True)[0].detach()
					v = torch.div(Wv, Wv.flatten().norm()+self.eps, out=v)
				if self.n_power_iterations > 0:
					# See above on why we need to clone
					v = v.clone(memory_format=torch.contiguous_format)

		# compute spectral norm (singular value) as ||W v||
		sigma = self.weight_fn(module, weight, v).flatten().norm()
		# rescale spectral norm
		sigma_val = sigma_range[0] if sigma_range[-1]==sigma_range[0] else sigma_range[0] + torch.sigmoid(sigma_var) * ( sigma_range[-1] - sigma_range[0] )
		return weight * (sigma_val/sigma)

	def remove(self, module: Module) -> None:
		with torch.no_grad():
			weight = self.compute_weight(module, do_power_iteration=False)
		delattr(module, self.name)
		delattr(module, self.name + '_orig')
		delattr(module, self.name + '_v')
		delattr(module, self.name + '_sigma_range')
		delattr(module, self.name + '_sigma_var')
		module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

	def __call__(self, module: Module, inputs: Any) -> None:
		# infer the input shape to create the buffer `v` for the singular vector if it doesn't exist
		with torch.no_grad():
			if getattr(module, self.name+'_v') is None:
				delattr(module, self.name+'_v')
				v = module.weight.new_empty(inputs[0].numel()//inputs[0].size(0)).normal_(0, 1)
				v = F.normalize(v, dim=0, eps=self.eps).reshape(1,*inputs[0].shape[1:])
				module.register_buffer(self.name+"_v", v)
		# normalize weight
		setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

	def weight_fn(self, module, weight, v):
		# weight function to use in compute_weight
		if isinstance(module, torch.nn.Linear):
			return F.linear(v, weight, bias=None)
		elif isinstance(module, torch.nn.Conv1d):
			return F.conv1d(v, weight, bias=None, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
		elif isinstance(module, torch.nn.Conv2d):
			return F.conv2d(v, weight, bias=None, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
		elif isinstance(module, torch.nn.Conv3d):
			return F.conv3d(v, weight, bias=None, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
		else:
			raise TypeError(f"`spectral_norm` is not implemented for the module {module}")

	@staticmethod
	def apply(module: Module, name: str, n_power_iterations: int, eps: float, sigma: Union[int,float,list]) -> 'SpectralNorm':
		for k, hook in module._forward_pre_hooks.items():
			if isinstance(hook, SpectralNorm) and hook.name == name:
				raise RuntimeError(f"Cannot register two spectral_norm hooks on the same parameter {name}")

		fn = SpectralNorm(name, n_power_iterations, eps)

		weight = module._parameters[name]
		if isinstance(weight, torch.nn.parameter.UninitializedParameter):
			raise ValueError(
				'The module passed to `SpectralNorm` can\'t have uninitialized parameters. '
				'Make sure to run the dummy forward before applying spectral normalization')

		# determine spectral norm range
		if isinstance(sigma, list):
			if len(sigma)==1:
				sigma_range = 3*sigma
			elif len(sigma)==2:
				sigma_range = [sigma[0],sigma[0],sigma[1]]
			elif len(sigma)==3:
				sigma_range = sigma
			else:
				raise ValueError(f"length of `sigma` list should be 1, 2 or 3, got {sigma}")
		elif isinstance(sigma, int) or isinstance(sigma, float):
			sigma_range = 3*[sigma]
		else:
			raise TypeError(f"`sigma` must be int, float, or list, got {type(sigma)}")
		# register parameters and buffers to handle learnable spectral norm
		ini_sigma   = torch.tensor( (sigma_range[1]-sigma_range[0]) / (sigma_range[2]-sigma_range[0]+1.e-8) + 0.01 )
		inv_sigmoid = torch.log(ini_sigma/(1-ini_sigma))
		module.register_parameter(fn.name + "_sigma_var", torch.nn.parameter.Parameter( inv_sigmoid, requires_grad=(sigma_range[-1]!=sigma_range[0])))
		module.register_buffer(fn.name + "_sigma_range", torch.tensor(sigma_range))

		delattr(module, fn.name)
		module.register_parameter(fn.name + "_orig", weight)
		# We still need to assign weight back as fn.name because all sorts of
		# things may assume that it exists, e.g., when initializing weights.
		# However, we can't directly assign as it could be an nn.Parameter and
		# gets added as a parameter. Instead, we register weight.data as a plain attribute.
		setattr(module, fn.name, weight.data)

		# singular vector is undefined before the first call because its shape is unknown
		module.register_buffer(fn.name + "_v", None)

		# pre_hook to calculate spectral norm every time `forward` is called
		module.register_forward_pre_hook(fn)
		# pre_hook to load the correct learned singular vector when loading the module,
		# note `with_module=True` option to get access to the module instance
		module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn), with_module=True)
		return fn


# This is a top level class because Py2 pickle doesn't like inner class nor an instancemethod
class SpectralNormLoadStateDictPreHook:
	def __init__(self, fn) -> None:
		r""" Register singular vector buffer before loading since it is undefined
		after `def apply()` above and will be reinitialized during the `__call__`
		"""
		self.fn = fn

	def __call__(self, module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
		fn = self.fn
		weight_key = prefix + fn.name
		if not weight_key+'_v' in state_dict:
			return
		else:
			module.register_buffer(fn.name+'_v', state_dict[weight_key+'_v'])


###############################################################################
###############################################################################


T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module,
				  name: str = 'weight',
				  n_power_iterations: int = 1,
				  eps: float = 1e-12,
				  sigma: Union[int,float,list] = 1.0) -> T_module:
	r"""Applies spectral normalization to a parameter in the given module.

	Spectral normalization rescales the weight tensor with spectral norm of the weight matrix calculated using
	power iteration method. This is implemented via a hook that calculates spectral norm and rescales weight
	before every `module.forward` call.

	Args:
		module (nn.Module): containing module
		name (str, optional): name of weight parameter
		n_power_iterations (int, optional): number of power iterations to calculate spectral norm
		eps (float, optional): epsilon for numerical stability in calculating norms
		sigma (int, float, or list, optional): range of admissible spectral norm values, defaults to 1
			expands to the list `[endpoint1,initial,endpoint2]` as follows
				value ->  [value,value,value]
				[value] -> [value,value,value]
				[initial,final] -> [initial,initial,final]

	Returns:
		The original module with the spectral norm hook
	"""
	if n_power_iterations>0:
		SpectralNorm.apply(module, name, n_power_iterations, eps, sigma)
	return module


def remove_spectral_norm(module: T_module, name: str = 'weight') -> T_module:
	r"""Removes the spectral normalization reparameterization from a module.

	Args:
		module (Module): containing module
		name (str, optional): name of weight parameter
	"""
	for k, hook in module._forward_pre_hooks.items():
		if isinstance(hook, SpectralNorm) and hook.name == name:
			hook.remove(module)
			del module._forward_pre_hooks[k]
			break
	else:
		raise ValueError(f"spectral_norm of '{name}' not found in {module}")

	return module



###############################################################################
###############################################################################


def eigenvalues(fun, x, type='jacobian', return_matrix=False):
    import numpy as np
    x = x.detach().requires_grad_(True)

    # import scipy.sparse.linalg as sla
    # from scipy.sparse.linalg import eigs, svds
    # # using scipy sparse LinearOperator
    # numpy_dtype = {torch.float: np.float, torch.float32: np.float32, torch.float64: np.float64}
    # # 'Matrix-vector' product of the linear operator
    # def matvec(v):
    #   v0 = torch.from_numpy(v).view_as(data).to(device=data.device, dtype=data.dtype)
    #   Av = torch.autograd.grad(self.forward(t,data), data, grad_outputs=v0, create_graph=False, retain_graph=True, only_inputs=True)[0]
    #   return Av.cpu().detach().numpy().ravel()
    # A_dot = sla.LinearOperator(dtype=numpy_dtype[data.dtype], shape=(data.numel(),data.numel()), matvec=matvec)
    #
    # # using scipy sparse matrix
    # A_dot  = jacobian( self.forward(t,data), data, True ).reshape( data.numel(),data.numel() ).cpu().detach().numpy()
    # eigvals  = eigs(A_dot, k=data.numel()-2, return_eigenvectors=False).reshape((-1,1))
    # singvals = svds(A_dot, k=5, return_singular_vectors=False)

    if isinstance(fun, torch.nn.Sequential) or isinstance(fun, torch.nn.ModuleList):
        eigvals = []
        for j in range(len(fun)):
            batch_dim = x.size(0)
            data_dim  = x.numel() // batch_dim
            y = 1*fun[j](x) # in case fun is just x
            if y.shape==x.shape or type=='hessian':
                if type=='jacobian':
                    jac = jacobian( y, x, True )
                elif type=='hessian':
                    jac = hessian( y, x, True )
                jac = jac.reshape( batch_dim, data_dim, batch_dim, data_dim ).cpu().detach().numpy()
                eig = np.linalg.eigvals([ jac[i,:,i,:] for i in range(batch_dim) ]).reshape((-1,1))
                eigvals.append( np.hstack(( np.real(eig), np.imag(eig) )) )
            else:
                eigvals.append(None)
            x = y
    else:
        batch_dim = x.size(0)
        data_dim  = x.numel() // batch_dim
        y = 1*fun(x) # in case fun is just x
        if y.shape==x.shape or type=='hessian':
            if type=='jacobian':
                jac = jacobian( y, x, True )
            elif type=='hessian':
                jac = hessian( y, x, True )
            jac = jac.reshape( batch_dim, data_dim, batch_dim, data_dim ).cpu().detach().numpy()
            eig = np.linalg.eigvals([ jac[i,:,i,:] for i in range(batch_dim) ]).reshape((-1,1))
            eigvals = np.hstack(( np.real(eig), np.imag(eig) ))
        else:
            eigvals = None
    if return_matrix:
        return eigvals, jac
    else:
        return eigvals




def spectralnorm(fun, x, power_iters=100):
    x = x.detach().requires_grad_(True)
    batch_dim = x.size(0)
    y = fun(x)
    v = torch.randn_like(y)
    for i in range(power_iters):
        with torch.enable_grad():
            v.requires_grad_(True)
            v1 = torch.autograd.grad(y, x, grad_outputs=v, create_graph=True, retain_graph=True, only_inputs=True)[0]
            v1 = 0.5*torch.autograd.grad(v1.pow(2).sum(), v, create_graph=False, retain_graph=False, only_inputs=True)[0]
        v.requires_grad_(False)
        v = torch.div(v1, v1.reshape((batch_dim,-1)).norm(dim=1,keepdim=True).reshape([batch_dim]+(y.ndim-1)*[1])+1.e-12, out=v)
    sigma = torch.autograd.grad(y, x, grad_outputs=v, create_graph=False, retain_graph=False, only_inputs=True)[0].reshape((batch_dim,-1)).norm(dim=1)
    return sigma
    # import numpy as np
    # import scipy.sparse.linalg as sla
    # from scipy.sparse.linalg import eigs, svds
    # if isinstance(fun, torch.nn.Sequential) or isinstance(fun, torch.nn.ModuleList):
    #     singvals = []
    #     for j in range(len(fun)):
    #         y = 1*fun[j](x) # in case fun is just x
    #         batch_dim = x.size(0)
    #         data_dim  = x.numel() // batch_dim
    #         out_dim   = y.numel() // batch_dim
    #         jac  = jacobian( y, x, True ).reshape( batch_dim, out_dim, batch_dim, data_dim ).cpu().detach().numpy()
    #         sing = np.linalg.svd([ jac[i,:,i,:] for i in range(batch_dim) ], compute_uv=False)
    #         if mode=='max':
    #             singvals.append(np.amax(sing,1))
    #         else:
    #             singvals.append(sing)
    #         x = y
    # else:
    #     y = 1*fun(x) # in case fun is just x
    #     batch_dim = x.size(0)
    #     data_dim  = x.numel() // batch_dim
    #     out_dim   = y.numel() // batch_dim
    #     jac = jacobian( y, x, True ).reshape( batch_dim, out_dim, batch_dim, data_dim  ).cpu().detach().numpy()
    #     singvals = np.linalg.svd([ jac[i,:,i,:] for i in range(batch_dim) ], compute_uv=False)
    #     if mode=='max':
    #         singvals = np.amax(singvals,1)
    # return singvals