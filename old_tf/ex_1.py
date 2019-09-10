import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense
import tensorflow.keras.backend as K
from CustomLayers import Antisym


import argparse



#########################################################################################
#########################################################################################



# Usage example:
# python ex_1.py --theta=0.5 --lr=0.01 --simulations=1 --layers=10 --step_size=0.1





#########################################################################################
#########################################################################################




parser = argparse.ArgumentParser()
parser.add_argument("--theta", type=np.float, default=0.5)
parser.add_argument("--lr", type=np.float, default=0.01)
parser.add_argument("--simulations", type=np.int, default=1)
parser.add_argument("--layers", type=np.int, default=10)
parser.add_argument("--step_size", type=np.float, default=0.1)
args = parser.parse_args()


theta = args.theta
learning_rate = args.lr
simulations = args.simulations
num_layers = args.layers
step_size = args.step_size

print("theta: "+str(theta))
print("learning_rate: "+str(learning_rate))
print("layers: "+str(num_layers))
print("step_size: "+str(step_size))
print("simulations: "+str(simulations))


K.clear_session(); sess = K.get_session()
K.set_image_data_format('channels_last')





#########################################################################################
#########################################################################################





# basic gradient descent nonlinear solver
def nsolve(functional,x0,maxiters=100):
	with tf.name_scope("nsolve"):
		lmbda = 1.e-5

		x = x0 - lmbda * tf.gradients(functional(x0),x0)[0]
		for i in range(maxiters):
			x = x - lmbda * tf.gradients(functional(x),x)[0]
		return x

# basic fixed point iteration solver
def fixed_point(F,x0,maxiters=100):
	with tf.name_scope("nsolve"):
		x = x0
		for i in range(maxiters):
			x = F(x)
		return x





#########################################################################################
#########################################################################################





class ImplicitNet_step(Model):
	'''
	Class implementing a single step of the implicit residual layer
	'''
	def __init__(self,units,h=1,theta=0.5):
		super(ImplicitNet_step, self).__init__()
		self.activation = Antisym(units=units,activation='relu')
		self.h     = h
		self.theta = theta

	def get_kernel(self):
		return self.activation.get_weights()[0]

	def StepFun(self, x):
		with tf.name_scope("FUNC"):
			y = self.activation(x)
			return self.h*y

	def call(self, x):
		if self.theta>0:
			x_shape  = tf.shape(x)
			Id  = tf.eye(x_shape[1],batch_shape=[x_shape[0]])
		def implicit_correction():
			@tf.custom_gradient
			def fun(y):
				def grad(dy):
					with tf.name_scope("Jacobian"):
						with tf.GradientTape() as g:
							g.watch(y)
							z = self.StepFun(y)
						Jac = g.batch_jacobian(z,y,experimental_use_pfor=False)
					operator = tf.linalg.LinearOperatorFullMatrix( Id - self.theta*Jac )
					return operator.solvevec(dy,adjoint=True)
				return y, grad
			return fun

		# step_function at previous time step
		with tf.name_scope("explicit"):
			F0 = self.StepFun(x)
			explicit = x + (1-self.theta)*F0

		if self.theta>0:
			reduction_axes = [ ax for ax in range(1,len(x.get_shape())) ]
			with tf.name_scope("linsolve"):
				with tf.name_scope("Jacobian"):
					with tf.GradientTape() as g:
						g.watch(x)
						z = self.StepFun(x)
					Jac = g.batch_jacobian(z,x)
				operator = tf.linalg.LinearOperatorFullMatrix( Id - self.theta*Jac )
				y_linear = x + operator.solvevec(F0,adjoint=False)
			# guessed nonlinear solution
			y = explicit + tf.stop_gradient(self.theta*self.StepFun(y_linear))

			with tf.name_scope("nsolve"):
				residual   = lambda z: z - explicit - self.theta*self.StepFun(z)
				functional = lambda z: tf.reduce_sum(K.square(residual(z)),axis=reduction_axes)
				y = nsolve(functional,x0=y,maxiters=20)
				y = fixed_point(lambda z:explicit+self.theta*self.StepFun(z),x0=y,maxiters=10)

			# residuals
			with tf.name_scope("residuals"):
				res      = residual(y)
				res_norm = tf.reduce_mean(K.sqrt(tf.reduce_sum(res*res,axis=reduction_axes)))
				tf.summary.scalar("residual",res_norm)

			y = explicit + tf.stop_gradient(self.theta*self.StepFun(y))

			return implicit_correction()(y)
		else:
			return explicit





class ODENet(Model):
	'''
	Class implementing an ODE solver
	'''
	def __init__(self,num_layers=10,num_units=4,h=1):
		super(ODENet,self).__init__()
		self.num_layers = num_layers
		self.num_units  = num_units

		self.step = []
		for t in range(self.num_layers):
			self.step.append( ImplicitNet_step(units=num_units,h=h,theta=theta) )

	def TV_regularizer(self):
		old_kernel = self.step[0].get_kernel()
		kernel = self.step[1].get_kernel()
		R = tf.reduce_sum( K.square( kernel - old_kernel ) )
		old_kernel = kernel
		for t in range(2,self.num_layers):
			kernel = self.step[t].get_kernel()
			R = R + tf.reduce_sum( K.square( kernel - old_kernel ) )
			old_kernel = kernel
		return R

	def call(self, x):
		y = Dense(units=self.num_units,activation='linear')(x)
		for t in range(self.num_layers):
			y = self.step[t](y)
		return y




#########################################################################################
#########################################################################################



npoints = 100
x = np.random.rand(npoints,1)*2-1
y = np.sin(2 * np.pi * x) * np.exp(-x**2 / 2)

nval = 200
xval = np.linspace(-1, 1, nval).reshape((nval,1))
yval = np.sin(2 * np.pi * xval) * np.exp(-xval**2 / 2)

data_shape = x[0].shape




#########################################################################################
#########################################################################################




def Loss(model):
	def loss(y_true,y_pred):
		return K.square(y_true-y_pred) + 0.1/model.num_layers*model.TV_regularizer()
	return loss

def Accuracy(y_true,y_pred):
	return K.square(y_true-y_pred)


ODE_model = ODENet(num_layers=num_layers,num_units=5,h=step_size)
inputs  = Input(shape=data_shape,dtype='float32')
outputs = ODE_model(inputs)
outputs = Dense(units=1,activation='linear',use_bias=True)(outputs)


optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
callback  = tf.keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=1,write_grads=False,update_freq='epoch')


model = Model(inputs=inputs, outputs=outputs)
ODE_model.summary()
model.summary()
model.compile(optimizer=optimizer,loss=Loss(ODE_model),metrics=[Accuracy])




#########################################################################################
#########################################################################################



for s in range(simulations):
	try:
		history = model.fit( x, y, batch_size=4, epochs=1000, callbacks=[callback], validation_data=(xval,yval))
	except:
		print("")
		print("Oops!!!!!!!!!!!!!!")
		print("")



#########################################################################################
#########################################################################################




ntest = 1000
xtest = np.linspace(-1, 1, ntest).reshape((ntest,1))
ytest = model.predict(xtest)
ytrue = np.sin(2 * np.pi * xtest) * np.exp(-xtest**2 / 2)

plt.plot(xtest,ytrue)
plt.plot(x,y,'o')
plt.plot(xtest,ytest)
plt.legend(('function','training data','prediction'),fontsize='x-large')
plt.show()