import neuralnetwork

nn = neuralnetwork.NeuralNetwork(2,4,1,0.3)

for n in range(10000):
	nn.train([1,1], [0])
	nn.train([1,0], [1])
	nn.train([0,1], [1])
	nn.train([0,0], [0])

print(nn.query([0,0]))
print(nn.query([0,1]))
print(nn.query([1,0]))
print(nn.query([1,1]))