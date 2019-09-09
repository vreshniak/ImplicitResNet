# ImplicitResNet

## Usage example:

python ex_3.py --lr 1.e-4 --layers 2 --filters 1 --epochs 500 --h 1.0 --batch_size 200 --theta 0.5 --seed 43

## Some notes on the implementation:

- scipy is used for nonlinear (CG) and linear (lgmres) solvers. However, explicit gradients for the CG solver and matrix-vector products for the linear solver are evaluated using torch.autograd capabilities (using GPU when available). __There is a potential for optimizing this part of the code__

- number of CG iterations is limited to some reasonable number so that nonlinear solver does not in fact converge to the true solution. It is not clear to me now if this is the right way to go
