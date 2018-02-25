import numpy as np
import scipy.special
class NeuralNetwork():
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inputnodes = inputnodes
		self.hiddennodes = hiddennodes
		self.outputnodes = outputnodes

		self.weightsinputhidden = np.random.normal(0.0, pow(self.hiddennodes, -0.5), (self.hiddennodes, self.inputnodes))
		self.weightshiddenoutput = np.random.normal(0.0, pow(self.outputnodes, -0.5), (self.outputnodes, self.hiddennodes))


		self.learningrate = learningrate

		self.activation_function = lambda x: scipy.special.expit(x)
		pass

	def train(self, inputlist, targetlist):
		inputs = np.array(inputlist, ndmin=2).T
		targets = np.array(targetlist, ndmin=2).T

		hidden_inputs = np.dot(self.weightsinputhidden, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.weightshiddenoutput, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs

		hidden_errors = np.dot(self.weightshiddenoutput.T, output_errors)

		self.weightshiddenoutput += self.learningrate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
		self.weightsinputhidden += self.learningrate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

		
		pass

	def query(self, inputslist):
		inputs = np.array(inputslist, ndmin=2).T
		hidden_inputs = np.dot(self.weightsinputhidden, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.weightshiddenoutput, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		return final_outputs
		pass