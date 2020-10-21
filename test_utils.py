import numpy as np

import torch
import utils


to_numpy = lambda x: x.detach().numpy()


##################################################
# define random function

dim = 4
sigma = torch.nn.ReLU()
F = torch.nn.Sequential(
		torch.nn.Linear(dim,dim), sigma,
		torch.nn.Linear(dim,dim), sigma,
		torch.nn.Linear(dim,dim), sigma,
		torch.nn.Linear(dim,dim) )
for name, weight in F.named_parameters():
	if 'weight' in name:
		torch.nn.init.uniform_(weight,-5.0,5.0)


##################################################


def test_TraceJacobianReg():
	n = 5000
	sims = 10
	divacc = 0
	jacacc = 0
	for _ in range(sims):
		x = torch.rand((1,dim)).requires_grad_(True)

		jacobian = utils.jacobian( 10*F(x), x, True ).reshape(dim,dim)
		divjac   = utils.TraceJacobianReg(n)
		div, jac = divjac(lambda x: 10*F(x), x)

		divacc = divacc + int(np.abs( to_numpy(torch.trace(jacobian)) - to_numpy(div) ) < 1.e-2)
		jacacc = jacacc + int(np.abs( to_numpy(jacobian.pow(2).sum()) - to_numpy(jac) ) < 1.e-2)

	assert divacc/sims >= 0.9
	assert jacacc/sims >= 0.9



divjac  = utils.TraceJacobianReg(1000)
jacdiag = utils.JacDiagReg(1000)
x = torch.rand((1,dim)).requires_grad_(True)
print( to_numpy(utils.jacobian( F(x), x, True ).reshape(dim,dim).diag().sum()) )
print( to_numpy(jacdiag(lambda x: F(x), x)) )
print( to_numpy(divjac(lambda x: F(x), x)[0]) )
exit()


A = np.random.rand(dim,dim)
A = A / np.linalg.norm(A) #/ np.linalg.norm(A)/ np.linalg.norm(A)
B = A
C = A
D = A
E = A
for _ in range(100):
	C = 0.5 * ( C + np.linalg.inv(C) )
	B = 0.5 * B @ ( 3*np.eye(dim) - B@B )
	# E = B @ ( 15*np.eye(dim) - 10*E@E + 3*E@E@E@E ) / 8
	D = 2.0 * D @ np.linalg.inv( np.eye(dim) + D@D )
print(np.linalg.eig(A)[0])
print(np.linalg.eig(C)[0])
print(np.linalg.eig(B)[0])
# print(np.linalg.eig(E)[0])
print(np.linalg.eig(D)[0])