import torch


def directional_derivative(fun, input, direction, create_graph=True):
	input  = input.detach().requires_grad_(True)
	output = fun(input)

	# v = direction.detach().requires_grad_(True)
	v = torch.ones_like(output).requires_grad_(True)

	# normalize direction
	batch_dim = direction.size(0)
	dir_norm  = torch.clamp( direction.reshape((batch_dim,-1)).norm(dim=1, keepdim=True), min=1.e-16 ) # avoid division by zero
	direction = direction / dir_norm

	grad_x = torch.autograd.grad(
		outputs=output,
		inputs=input,
		grad_outputs=v,
		create_graph=True)[0]

	grad_v, = torch.autograd.grad(
		outputs=grad_x,
		inputs=v,
		grad_outputs=direction.detach(),
		create_graph=create_graph)  # need create_graph to find it's derivative

	return grad_v



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

def jacobian(output, input, create_graph=False):
	jacobian = []

	Id = torch.zeros(*output.shape).to(input.device)
	for i in range(output.numel()):
		Id.data.flatten()[i] = 1.0

		jac_i = torch.autograd.grad(
			outputs=output,
			inputs=input,
			grad_outputs=Id,
			create_graph=create_graph,  # need create_graph to find it's derivative
			only_inputs=True)[0]

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