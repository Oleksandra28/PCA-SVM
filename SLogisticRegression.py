__author__ = 'Oleksandra'

import math

import numpy as np

class SLogisticRegression():

    def __init__(self):
        self.initial_theta = np.array([])
        self.theta = np.array([])
#------------------------------------------------------------------------------------------------------------------
    def sigmoid(self, z):
        return 1./(1 + math.e**(-z))
#------------------------------------------------------------------------------------------------------------------
    def fit(self, features_train, labels_train):
        # set all parameters theta to be equal to 1
        # add intercept term '+1'
        self.initial_theta.resize(len(features_train) + 1)
        self.initial_theta.fill(1)
        print 'self.initial_theta dimention : ', self.initial_theta.shape
        z = self.initial_theta.T*self.initial_theta
        print 'self.initial_theta transposed dimention : ', self.initial_theta.T.T.shape


        #z = zip(*features_train)*self.initial_theta
        print 'z : ', z
        initial_predictions = self.sigmoid(z)
#------------------------------------------------------------------------------------------------------------------
    def predict(self, features):
        pass
        #z = self.theta*features
        #self.sigmoid(z)
#------------------------------------------------------------------------------------------------------------------

