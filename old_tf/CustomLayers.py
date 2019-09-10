import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K


class Antisym(Dense):
	def __init__(self,units,activation='relu'):
		super(Antisym,self).__init__(units,activation=activation)

	def call(self, inputs):
		return self.activation( tf.linalg.matmul(inputs,self.kernel-K.transpose(self.kernel)) + self.bias )
