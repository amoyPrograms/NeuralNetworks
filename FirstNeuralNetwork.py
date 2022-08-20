 
import numpy as np
 

class NeuralNetwork():
    def __init__(self, input, output):
        self.input = input
        self.output= output
        self.weight0 = 2 * np.random.random((4,1)) - 1
        self.weight1 = 2 * np.random.random((4,1)) - 1
        #self.bias = np.random.randn()
        
        #print (self.weight0)
        
    def sigmond(self, input, deriv = False):
        if deriv == True:
            return input * (1-input)
        return 1/(1+np.exp(-input))

    def forward(self):
        self.input_layer = input
        l1 = np.dot(self.input_layer, self.weight0)
        self.layer1 = self.sigmond(l1)


        l2 = np.dot(self.layer1.T, self.weight1) 
        self.layer2 = self.sigmond(l2)
     
    def back(self):
        error1= output - self.layer1
        delta_1 = error1 * self.sigmond(self.layer1, True)

        error2= self.layer1 - self.layer2
        delta_2 = error2 * self.sigmond(self.layer2, True)
        #Update the weight
        self.weight1 = self.weight1 + np.dot(self.layer1.T, delta_2)
        #print("Delta 2: ", delta_2.shape)
        #print("error 2: ", error2.shape)
        self.weight0 = self.weight0 + np.dot(self.input_layer.T, delta_1)
       

input = np.array([
                    [1,1,0,0],
                    [0,1,0,0], 
                    [1,1,1,0], 
                    [1,0,0,1],

                ])
        
output = np.array([[1,0,1,0]]).T
 
Network = NeuralNetwork(input, output)

for epoch in range(1000):
    Network.forward()
    Network.back()
    if epoch % 100 ==0:
        print("Aim:{} Calculation: {} " .format(Network.output.T, Network.layer1.T ))
        #print("layer 1: ", NN.layer1)
        #print("layer 2: ", NN.layer2)
        
        pass
  
  
