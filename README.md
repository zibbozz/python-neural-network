# Neuronales Netzwerk in Python

## Implementierung des Netzwerks

~~~python
import neuralnetwork

nn = neuralnetwork.NeuralNetwork(<InputNeuronen>, <HiddenNeuronen>, <OutputNeuronen>, <Lernrate>)

nn.train([<Inputdaten>], [<Erwartete Ergebnisse>])

print(nn.query([<Inputdaten>]))
~~~