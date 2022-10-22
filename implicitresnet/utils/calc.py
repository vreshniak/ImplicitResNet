import torch



###############################################################################


def Fv(F, v):
	'''
	Compute `v` component of `F`, i.e., `Fv = (F,v/|v|)`
	'''
	if F.shape!=v.shape:
		raise ValueError(f"`F` and `v` must have the same shape, got F.shape={F.shape} and v.shape={v.shape}")
	batch_dim = F.shape[0]
	# unit vector in the given direction `v`
	v = v.detach()
	v = torch.nn.functional.normalize(v.reshape(batch_dim,-1), p=2, dim=1)
	F = F.reshape(batch_dim,-1)
	return (F*v).sum(axis=1, keepdim=False)



###############################################################################
# derivatives


def jacT_dot_v(output, input, v, create_graph=True):
	'''
	Jacobian^T vector product `(d output / d input)^T @ v`
	'''
	return torch.autograd.grad(
		outputs=output,
		inputs=input,
		grad_outputs=v,
		create_graph=create_graph, # need to create_graph to find it's derivative
		only_inputs=True)[0]


def jac_dot_v(output, input, v, create_graph=True):
	'''
	Jacobian vector product `(d output / d input) @ v`
	'''
	# dummy vector to compute Jacobian instead of transpose
	dummy = torch.ones_like(output).requires_grad_(True)

	# J^T @ dummy
	JT_dummy = jacT_dot_v(output, input, dummy, create_graph=True)

	# (J^T)^T @ v = J @ v
	J_v = jacT_dot_v(JT_dummy, dummy, v, create_graph=create_graph)

	return J_v


def directional_derivative(F, input, direction, normalize_direction=False, create_graph=True):
	'''
	Compute directional derivative of the function, i.e., `dF/dv = grad(F) @ v`
	'''
	batch_dim = input.size(0)
	if input.shape!=direction.shape:
		raise ValueError(f"Direction vector must have the same shape as the input, got direction.shape={direction.shape}, input.shape={input.shape}")

	if callable(F):
		input  = input.detach().requires_grad_(True)
		output = F(input)
	else:
		output = F

	if normalize_direction:
		direction = torch.nn.functional.normalize(direction.reshape(batch_dim,-1), p=2, dim=1).reshape(input.shape)

	return jac_dot_v(output, input, direction, create_graph)


def dFv_dv(F, input, v, create_graph=True):
	'''
	Compute directional derivative `dFv/dv = grad(Fv) @ v` of the `Fv` component of the vector field `F`
	'''
	batch_dim = input.size(0)
	if input.shape!=v.shape:
		raise ValueError(f"Direction vector `v` must have the same shape as the input, got direction.shape={v.shape}, input.shape={input.shape}")

	if callable(F):
		input  = input.detach().requires_grad_(True)
		output = F(input)
	else:
		output = F

	# `v` component of `F`
	F_v = Fv(output,v).unsqueeze(1)

	return directional_derivative(F_v, input, v, normalize_direction=False, create_graph=create_graph)


def gradient(F, input, create_graph=False, normalize=True, p=2):
	'''
	Compute gradient of `F`
	'''
	batch_dim = input.size(0)

	if callable(F):
		input  = input.detach().requires_grad_(True)
		output = F(input)
	else:
		output = F
	if output.dim()!=1:
		raise ValueError(f"`F` must be scalar valued, i.e., have only batch dimension, got F.shape[1]={output.shape[1]}")

	gradF, = torch.autograd.grad(
		outputs=output,
		inputs=input,
		grad_outputs=torch.ones(batch_dim),
		create_graph=create_graph,  # need create_graph to find it's derivative
		only_inputs=True)
	if normalize:
		gradF = torch.nn.functional.normalize(gradF.reshape(batch_dim,-1),p=p,dim=1).reshape(input.shape)

	return gradF


def grad_norm_2(F, input, create_graph=False, normalize=True, p=2):
	'''
	Compute gradient of the squared 2-norm of `F`, i.e., `grad |F|^2`
	'''
	batch_dim = input.size(0)

	if callable(F):
		input  = input.detach().requires_grad_(True)
		output = F(input)
	else:
		output = F
	output = output.reshape(batch_dim,-1).pow(2).sum(dim=1)
	return gradient(output, input, create_graph, normalize, p)


def jacobian(F, input, create_graph=False):
	'''
	Compute exact Jacobian of `F`, i.e., `dF/dinput`
	Evaluation is performed elementwise and is quite inefficient
	'''
	jacobian = []

	if callable(F):
		input  = input.detach().requires_grad_(True)
		output = F(input)
	else:
		output = F

	Id = torch.zeros(*output.shape).to(input.device)
	for i in range(output.numel()):
		Id.data.flatten()[i] = 1.0

		jac_i, = torch.autograd.grad(
			outputs=output,
			inputs=input,
			grad_outputs=Id,
			create_graph=create_graph,  # need create_graph to find it's derivative
			only_inputs=True)

		jacobian.append(jac_i)

		Id.data.flatten()[i] = 0.0
	return torch.stack(jacobian, dim=0).reshape(output.shape+input.shape)


def hessian(output, input, create_graph=False):
	# assert output.size(1)==1, "output must be scalar function, got output.size(1)="+str(output.size(1))

	# gradient of output
	output_grad = torch.autograd.grad(
		outputs=output,
		inputs=input,
		create_graph=True,  # need create_graph to find it's derivative
		only_inputs=True)[0]

	hessian = jacobian(output_grad, input, create_graph)
	return hessian


###############################################################################
# Stochastic estimators
#
# References:
# 	1) Bekas, C., Kokiopoulou, E., Saad, Y.: An estimator for the diagonal of a matrix. Appl. Numer. Math. 57(11), 1214–1229 (2007)


@torch.enable_grad()
def jacobian_frobenius_norm_2(F, input, create_graph=True, n=1, rnd='rademacher'):
	'''
	Compute squared Frobenius norm of the Jacobian of `F` at `input`
	'''
	batch_dim = input.shape[0]

	if callable(F):
		input  = input.detach().requires_grad_(True)
		output = F(input)
	else:
		output = F

	t = q = 0
	if rnd=='rademacher':
		for _ in range(n):
			v  = 2 * (torch.rand_like(output)<0.5) - 1
			Jv = jacT_dot_v(output, input, v)
			t  = t + Jv*Jv
		q = n
	elif rnd=='gaussian':
		for _ in range(n):
			v  = torch.randn_like(output)
			Jv = jacT_dot_v(output, input, v)
			t  = t + Jv*Jv
			q  = q + v*v
		q = q + 1.e-12
	return (t / q).reshape((batch_dim,-1)).sum(axis=1)


def trace_and_jacobian(fun, input, create_graph=True, n=1, min_eig=None):
	'''
	Compute trace and Frobenius norm of the Jacobian averaged over the batch dimension
	'''
	with torch.enable_grad():
		batch_dim = input.size(0)

		input  = input.detach().requires_grad_(True)
		output = fun(input)

		div = jac = t = q = 0
		if hasattr(fun, 'trace'):
			div = fun.trace(input).sum() * n

		# Randomized version based on Hutchinson algorithm for trace estimation
		for _ in range(n):
			v = torch.randn_like(output)
			v_jac, = torch.autograd.grad(
				outputs=output,
				inputs=input,
				grad_outputs=v,
				create_graph=create_graph,  # need create_graph to find it's derivative
				only_inputs=True)
			vjacv = v*v_jac
			if not hasattr(fun, 'trace'):
				# if min_eig is not None:
				# 	# w = torch.clamp(vjacv - min_eig, min=0.0).detach()
				# 	# w = (w / torch.max(w)) #.pow(0.2)
				# 	w = torch.heaviside(vjacv/(v*v+1.e-12)-min_eig,torch.tensor([0.0])).detach()
				# 	# w = torch.nn.functional.leaky_relu( vjacv/(v*v) - min_eig, 0.5 ).detach()
				# 	# w = torch.nn.functional.relu(vjacv/(v*v) - min_eig).detach()
				# 	# w = (vjacv/(v*v+1.e-12) - min_eig).detach()
				# 	# w = w / (torch.max(w.abs())+1.e-6)
				# 	# w = torch.nn.functional.relu(w / (torch.max(w.abs())+1.e-6), inplace=True)**0.5
				# 	div = div + (w*vjacv).sum()
				# 	# div = div + (w*vjacv).sum()
				# 	# div = div*(w.numel()/torch.count_nonzero(w))
				# else:
				# 	div = div + vjacv.sum()
				div = div + vjacv.sum()
			jac = jac + v_jac.pow(2).sum()

	return div/n/batch_dim, jac/n/batch_dim #, torch.sum(w) #torch.sum(w)<0.8*w.numel()


def jacobian_diag(fun, input, create_graph=True, n=1):
	'''
	Compute stochastic estimate of the Jacobian diagonal:
	Bekas, C., Kokiopoulou, E., Saad, Y.: An estimator for the diagonal of a matrix. Appl. Numer. Math. 57(11), 1214–1229 (2007)
	'''
	with torch.enable_grad():
		input  = input.detach().requires_grad_(True)
		output = fun(input)

		t = q = 0
		for _ in range(n):
			# v = torch.randn_like(output)
			v = (torch.rand_like(output)<0.5)*2-1 # Rademacher
			v_jac, = torch.autograd.grad(
				outputs=output,
				inputs=input,
				grad_outputs=v,
				create_graph=create_graph,  # need create_graph to find it's derivative
				only_inputs=True)
			t = t + v*v_jac
			# q = q + v*v
	return t #/ (q+1.e-12)




def partial_trace(fun, input, create_graph=True, n=1, fraction=1.0):
	'''
	Compute stochastic estimate of the Jacobian diagonal:
	Bekas, C., Kokiopoulou, E., Saad, Y.: An estimator for the diagonal of a matrix. Appl. Numer. Math. 57(11), 1214–1229 (2007)
	'''
	with torch.enable_grad():
		batch_dim = input.size(0)

		input  = input.detach().requires_grad_(True)
		output = fun(input)

		t = 0
		q = 0
		for _ in range(n):
			v = torch.randn_like(output)
			v_jac, = torch.autograd.grad(
				outputs=output,
				inputs=input,
				grad_outputs=v,
				create_graph=create_graph,  # need create_graph to find it's derivative
				only_inputs=True)
			t = t + v*v_jac
			q = q + v*v
		diag = t / q
		# print(q)
		# div = t.sum()
		t = t.reshape((batch_dim,-1))
		numel = int(fraction * t.size(1))
		div = t[:,:numel].sum()

	return div/n/batch_dim, 0


class TraceJacobianReg(torch.nn.Module):
	def __init__(self, n=1):
		super().__init__()
		self.n = n

	def forward(self, fun, input, create_graph=True):
		'''
		Compute trace and Frobenius norm of the Jacobian averaged over the batch dimension
		'''
		with torch.enable_grad():
			batch_dim = input.size(0)

			input  = input.detach().requires_grad_(True)
			output = fun(input)

			div = jac = t = q = 0
			if hasattr(fun, 'trace'):
				div = fun.trace(input).sum() * self.n

			# Randomized version based on Hutchinson algorithm for trace estimation
			for _ in range(self.n):
				v = torch.randn_like(output)
				v_jac, = torch.autograd.grad(
					outputs=output,
					inputs=input,
					grad_outputs=v,
					create_graph=create_graph,  # need create_graph to find it's derivative
					only_inputs=True)
				vjacv = v*v_jac
				if not hasattr(fun, 'trace'):
					div = div + vjacv.sum()
				jac = jac + v_jac.pow(2).sum()
				# t = t + vjacv
				# q = q + v*v

			# remove diagonal elements
			# jac = (jac/self.n-(t/q).pow(2).sum())/batch_dim

		return div/self.n/batch_dim, jac/self.n/batch_dim
# class TraceJacobianReg(torch.nn.Module):
# 	def __init__(self, n=1, div=True, jac=False):
# 		super().__init__()
# 		self.n = n
# 		self.is_div = div
# 		self.is_jac = jac

# 	def forward(self, output, input, create_graph=True):
# 		'''
# 		Compute Frobenius norm of the Jacobian averaged over batch dimension
# 		'''
# 		batch_dim = input.size()[0]

# 		div = jac = 0
# 		if not self.is_div and not self.is_jac:
# 			return div, jac

# 		# Randomized version based on Hutchinson algorithm for trace estimation
# 		for _ in range(self.n):
# 			v = torch.randn_like(output)
# 			v_jac, = torch.autograd.grad(
# 				outputs=output,
# 				inputs=input,
# 				grad_outputs=v,
# 				create_graph=create_graph,  # need create_graph to find it's derivative
# 				only_inputs=True)
# 			if self.is_div: div = div + (v*v_jac).sum()
# 			if self.is_jac: jac = jac + v_jac.pow(2).sum()

# 		return div/self.n/batch_dim, jac/self.n/batch_dim



class JacDiagReg(torch.nn.Module):
	def __init__(self, n=1, value=None):
		super().__init__()
		self.n = n
		self.value = value

	def forward(self, fun, input, create_graph=True):
		'''
		Compute stochastic estimate of the Jacobian diagonal:
		Bekas, C., Kokiopoulou, E., Saad, Y.: An estimator for the diagonal of a matrix. Appl. Numer. Math. 57(11), 1214–1229 (2007)
		'''
		batch_dim = input.size(0)

		input  = input.detach().requires_grad_(True)
		output = fun(input)

		t = 0
		q = 0
		for _ in range(self.n):
			v = torch.randn_like(output)
			v_jac, = torch.autograd.grad(
				outputs=output,
				inputs=input,
				grad_outputs=v,
				create_graph=create_graph,  # need create_graph to find it's derivative
				only_inputs=True)
			t = t + v*v_jac
			q = q + v*v

		if self.value is not None:
			return ((t/q)-self.value).pow(2).sum() / batch_dim
		else:
			return t / q
			# return (t / q).reshape((batch_dim,-1))






###############################################################################

# def divergence(output, input, create_graph=False):
# 	jac = jacobian(output, input, create_graph).reshape((output.shape[0],output.numel()//output.shape[0],input.shape[0],input.numel()//input.shape[0]))
# 	return torch.stack([ jac[i,:,i,:].diag().sum() for i in range(output.shape[0]) ], dim=0)



# class divreg(torch.nn.Module):
# 	def __init__(self, n=1):
# 		self.n = n
# 		super(divreg, self).__init__()

# 	def forward(self, output, input, create_graph=False):
# 		reg = 0
# 		for i in range(self.n):
# 			v = torch.randn_like(input)
# 			jac_v, = torch.autograd.grad(
# 				outputs=output,
# 				inputs=input,
# 				grad_outputs=v,
# 				create_graph=create_graph,  # need create_graph to find it's derivative
# 				only_inputs=True)
# 			reg = reg + (v*jac_v).sum()
# 		return reg / self.n / input.size()[0]




# def jacobian_diag(output, input, create_graph=False):
# 	jacobian = []

# 	assert output.numel()==input.numel(), "input and output must have the sane shape"

# 	# out = output.flatten()
# 	# inp = input.flatten()
# 	# print(out[0])
# 	# print(inp[0])

# 	# for i in range(input.numel()):
# 	for i, j in zip(output, input):
# 		print(i,j)
# 		exit()
# 		jac_i = torch.autograd.grad(
# 			outputs=out[i],
# 			inputs=inp[i],
# 			grad_outputs=None,
# 			create_graph=create_graph,  # need create_graph to find it's derivative
# 			only_inputs=True)[0]
# 		jacobian.append(jac_i)

# 	print(jac_i)
# 	exit()

# 	return torch.stack(jacobian, dim=0)