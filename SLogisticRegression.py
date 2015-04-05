__author__ = 'Oleksandra'

import numpy as np
from SPCA import SPCA



class SLogisticRegression():

    def __init__(self):
        self.sigmoid = np.vectorize(self.sigmoid)

#------------------------------------------------------------------------------------------------------------------
    def sigmoid(self, z):
        return 1./(1 + np.exp(-z))
#------------------------------------------------------------------------------------------------------------------
    def fit(self, thetas):
        print 'given thetas dimensions : ', thetas.shape
        self.thetas = thetas
        #self.thetas = np.transpose(thetas)
        #print 'thetas transposed dimensions : ', self.thetas.shape
#------------------------------------------------------------------------------------------------------------------
    def predict(self, features):
        z = np.dot(features, self.thetas)
        print 'z dimensions : ', z.shape
        prediction = self.sigmoid(z)
        print 'prediction : ', prediction.shape
        return prediction
#------------------------------------------------------------------------------------------------------------------

