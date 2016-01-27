from collections import deque
import random
import math
import sys
class NeuronLayers(object):
    """docstring for NeuronLayer"""
    def __init__(self):
        super(NeuronLayer, self).__init__()
    def getConnexionByLayer(self,layernumber):
        pass
    def getOutputLayer(self):
        pass
    def getInputLayer(self):
        pass
    def getAllLayersWithoutInput(self):
        pass
    def getAllLayersWithoutOutput(self):
        pass
    def getAllLayersWithoutOutputAndInput(self):
        pass
class NeuronLayer(object):
    """docstring for NeuronFarm"""
    def __init__(self,layer):
        super(NeuronFarm, self).__init__()
    def getBiasNeurons(self):
        pass
    def getAllNeurons(self):
        pass
    def getNeuron(self,index):
        pass


class Net(object):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.layers = []
        self.error = 0.0
        self.recentAverageError = 0.0
        self.recentAverageErrorSmoothingFactor = 0.5
    def init(self,layers):
        #creates all the layers and adds the neurons to the layer
        for i,nbNeuron in enumerate(layers):
            print "creating layer {} with {} neurons".format(i,nbNeuron)
            l = Layer(i)
            num_outputs = 0 if len(layers)-1 == i else layers[i+1]
            for neuronNum in xrange(nbNeuron):
                n = Neuron(num_outputs,neuronNum)
                l.addNeuron(n)
            self.addLayer(l)
    def save(self):
        pass
    def populate(self):
        pass
    def addLayer(self,l):
        self.layers.append(l)
    def getResults(self):
        _res = []
        #get outputs from neurons except bias neurons
        for n in self.layers[-1].neurons[:-1]:
            _res.append(n,getOutputValue())
    def backPropagate(self,targetData):
        #calculate error RMS
        outputLayer = self.layers[-1]
        error = 0.0
        for i,n in enumerate(outputLayer.neurons):
            delta =  targetData[i] - n.getOutputValue()
            error = delta * delta
        error /= len(outputLayer.neurons) - 1
        error = math.sqrt(error)

        #average measurement
        recentAvergeError = self.recentAvergeError
        recentAvergeErrorSmoothingFactor = self.recentAvergeErrorSmoothingFactor
        recentAverageError = (recentAverageError * recentAvergeErrorSmoothingFactor + error) / (recentAverageErrorSmoothingFactor + 1.0 )
        self.recentAverageError = recentAverageError
        #calculate gradients without taking the bias node
        for i,n in enumerate(outputLayer.neurons[:-1]):
            n.calcOutputGradients(targetData[i])
        for i,l in enumerate(self.layers[1:-1]):
            hiddenLayer = l
            nextLayer = self.layers[i+1]
            for n in l.neurons:
                n.calcHiddenGradients(nextLayer)
        #update connexion weights
        for i,l in enumerate(self.layers[1:]):
            prevLayer = self.layers[i-1]
            for n in l.neurons:
                n.updateInputWeights(prevLayer)
    def feedForward(self,data):
        if len(data) == len(self.layers[0]):
            #set input values
            for n in self.layers[0].neurons:
                n.setOutputValue(data)
            prevLayer = None
            print [l.id for l in self.layers[-2::-1]]
            for i,l in enumerate(self.layers[-2::-1]):
                prevLayer = self.layers[i-1]
                for n in l.neurons:
                    n.feedForward(prevLayer)
            # for layerNum in xrange(self.layers):
            #     if layerNum is not 0: #propagation
            #         prevLayer = self.layers[layerNum - 1]
            #         for numNeuron in xrange(net.layers[layerNum]):
            #             self.layers[layerNum][numNeuron].feedForward(prevLayer)
        else:
            print "The training data doesnt have as much input as the input layer of the network"
class Layer(object):
    """docstring for Layer"""
    def __init__(self,layerNumber):
        super(Layer, self).__init__()
        self.neurons = []
        self.id = layerNumber
    def __iter__(self):
        return self.neurons
    def __len__(self):
        return len(self.neurons)
    def addNeuron(self,n):
        self.neurons.append(n)

class Neuron(object):
    """docstring for Neuron"""
    eta = 0.2 #overal net training rate
    alpha = 0.5 #multiploer of the last weight change
    def __init__(self,numOutputs, index):
        super(Neuron, self).__init__()
        print "making a neuron {}, with {} outputs".format(index,numOutputs)
        self.outputValue = None
        self.outputWeights = [] #vector of connections
        self.numOutputs = numOutputs
        self.index = index
        self.gradient = None
        for c in xrange(numOutputs):
            c = Connection()
            print "creating a connection to this neuron"
            self.outputWeights.append(Connection())
    def __str__(self):
        return ""
    def getOutputValue(self):
        return self.outputValue
    def setOutputValue(self,val):
        self.outputValue = val
    def getOutputWeights(self):
        return self.outputWeights
    def feedForward(self,layer):
        _sum = 0.0
        print layer.id
        for n in layer.neurons:
            print n.getOutputWeights()
            print n.index
            print self.index
            _sum += n.getOutputValue() * \
                    n.getOutputWeights()[self.index].weight
        #output value is the output with the wights of the previous layer of neurons
        self.outputValue = Neuron.transferFunction(_sum)
    def calcOutputGradients(self,value):
        delta = value - self.outputValue
        self.gradient = delta * Neuron.transferFunctionDerivitive(self.outputValue)
    def sumDOW(self,layer):
        _sum = 0.0
        for n in layer.neurons[:-1]:
            _sum += n.outputWeights[n.index].weight * n.gradient
        return _sum
    def calcHiddenGradients(self,layer):
        dow = self.sumDOW(layer)
        self.gradient = dow * Neuron.transferFunctionDerivitive(self.outputValue)
    def updateInputWeights(self,layer):
        #modify the previous layer
        for n in layer.neurons:
            oldDeltaWeight = n.outputWeights[self.index].deltaWeight
            newDeltaWeight = \
                self.eta \
                * n.getOutputValue() \
                * self.gradient \
                + self.alpha \
                * oldDeltaWeight
            n.outputWeights[self.index].deltaWeight = newDeltaWeight
            n.outputWeights[self.index].weight += newDeltaWeight
    @staticmethod
    def transferFunction(x):
        """Tanh """
        return math.tanh(x)
    @staticmethod
    def transferFunctionDerivitive(x):
        """Reverse tanh"""
        return 1 - x * x
class Connection(object):
    """docstring for Connection"""
    weight = None
    deltaWeight = None
    MAXRAND = 1
    def __init__(self):
        super(Connection, self).__init__()
        self.weight = random.random() / self.MAXRAND

class Brain(object):
    """Brain class, handles all the calls to the neural network
    This class handles each call to the Net class with more meaningfull functions
    TODO: Handle the network in a seperate thread
    """
    def __init__(self):
        super(Brain, self).__init__()
        self.training = None
        self.targets = None
    def init(self,layers):
        network = Net()
        network.init(layers)
        self.network = network
    def setTargets(self,targets):
        self.targets = targets
    def setData(self,data):
        self.training = data
    def simulate(self):
        while self.network.getResults() not in self.targets:
            self.train()
    def feed(self,trainingData):
        self.network.feedForward(trainingData)
    def sendData(self,data):
        self.network.feedForward(data)
    def train(self):
        for i,foo in enumerate(self.targets):
            #first we feed it the data
            self.feed(self.training[i])
            #now we adapt the nets output
            self.network.backPropagate(self.targets[i])
    def getResults(self):
        return self.network.getResults()
class NetworkTopology(object):
    layers = []
    """docstring for NetworkTopology"""
    def __init__(self):
        super(NetworkTopology, self).__init__()
        self.layers = []
    def getLayers(self):
        return self.layers
    def addLayer(self,neurons):
        self.layers.append(neurons)
def main():
    b = Brain()
    topology = NetworkTopology()
    # num_layers = int(raw_input("Number of Layers ? > "))
    # for l in xrange(num_layers):
    #     num_neurons = raw_input("Number of neurons for layer {} (layer 0 is input layer, and layer {} is the last layer) ? > ".format(l,num_layers-1))
    #     topology.addLayer(int(num_neurons))
    topo = [2,4,4,1] #inforce a topology
    b.init(topo)
    #f = raw_input("Where is the data to train our braaaaiin ? > ")
    f = "xorData.txt"
    data = open(f,"r").read().split("\n")
    training_data = []
    targets = []
    for line in data:
        t = line.split(":")
        if len(t) < 2:break
        # a line should be "trainingvalues,trainingvalues,trainingvalues,trainingvalues,trainingvalues:target data,target data"
        # so the first part before the : is the training data, and after that it's the target data
        # t[0] => training data
        # t[1] => target values for the training data
        training_data.append([int(x.strip(" \t\n\r")) for x in t[0].split(',')])
        targets.append([int(x.strip(" \t\n\r")) for x in t[1].split(',')])
    b.setData(training_data)
    b.setTargets(targets)
    b.train()
    try:
        while True:
            data = raw_input("Wanna try sending some data (comma seperated values) ? >")
            b.sendData([x.strip() for x in data.split(',')])
            print "output of this is {}".format(b.getResults())
    except KeyboardInterrupt:
        sys.exit(0)
if __name__ == '__main__':
    main()
