import numpy as np
import pandas as pd

class NeuralNetwork():
    def __init__(self, input, output):
        self.input = input
        self.output= output
        self.weight1 = np.random.random((input.shape[1],1)) * 0.1
        self.bias1 = np.random.random((13,1)) 

        print ("Weight: " , self.weight1)
       
    def sigmond(self, input, deriv = False):
        if deriv == True:
            return input * (1-input)
        return 1/(1+np.exp(-input))
 
    def forward(self):
        self.input_layer = input

    
        l1 = np.dot(self.input_layer, self.weight1) + self.bias1
        self.layer1 = self.sigmond(l1)

     
    def back(self):
        error1= output - self.layer1
        delta_1 = error1 * self.sigmond(self.layer1, True)

        self.weight1 = self.weight1 + np.dot(self.input_layer.T, delta_1)


 
input = np.array([
                    [16,21,2,26,82],
                    [19,53,28,17,20],
                    [25,18,17,12,28],
                    [30,21,13,20,40],
                    [33,31,20,4,17],
                    [10,22,19,7,53],
                    [52,24,5,3,70],
                    [33,22,21,42,32],
                    [61,37,1,44,71],
                    [11,62,50,23,14],
                    [16,30,51,19,23],
                    [54,81,71,12,24],
                    [37,58,91,12,2],
                ])/260
       
output = np.array([[1,1,0,0,1,1,1,0,1,0,0,1,0]]).T
 
NN = NeuralNetwork(input, output)
 
for epoch in range(7000): 
    NN.forward()
    NN.back()
    if epoch % 100==0:
        print("Aim:{} Calculation: {} " .format(NN.output.T, NN.layer1.T ))

        print(input.shape[1])
     
       
        pass
 
  
  
