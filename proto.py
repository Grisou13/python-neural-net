class Network:
    def __init__(self):
        pass
class Neuron:
    def __init__(self):
        pass
    def bindToNeuron(self,neuron):
        pass
    def bindToLayer(self,layerId):
        pass
    def getOutputValue(self):
        pass
    def setOutputValue(self,value):
        pass
    def getOutputWeight(self):
        pass
    def updateConnectionWeights(self,weight,deltaWeight):
        pass
    @staticmethod
    def transferFunction(x):
        """Tanh """
        return math.tanh(x)
    @staticmethod
    def transferFunctionDerivitive(x):
        """Reverse tanh"""
        return 1 - x * x
class Layer:
    def __init__(self):
        pass
    def addNeuron(self):
        pass
    def removeNeuron(self):
        pass

class NeuronConnection:
    weight = None
    deltaWeight = None
    MAXRAND = 1
    def __init__(self):
        super(Connection, self).__init__()
        self.weight = random.random() / self.MAXRAND #create a new connection with a random weight
net = Network()
net.setTopology(LayerTopology)
net.train(TrainingData,trainingTargets)
net.ask(Data)

feed data to input:
    for each neuron
        neuron.getOutputValue
        next neuron.setInputValue()
