__author__ = 'osopova'

from SPCA import SPCA

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from SLogisticRegression import SLogisticRegression
import numpy as np

class ClassificationManager():

    def __init__(self, dataset, cross_validation_percent = 0.2, test_percent = 0.2):

        self.test_percent = test_percent

        self.data = dataset.data
        print 'dataset data dimensions : ', self.data.shape
        labels = dataset.target

        features_train, features_test, labels_train, labels_test = train_test_split(self.data, labels, test_size = self.test_percent)

        self.pca = SPCA()

        # PCA for training data
        self.pca.fit(features_train)
        print 'train features dimensions before PCA : ', features_train.shape
        features_train_PCA = self.pca.transform(features_train)
        print 'train features dimensions after PCA : ', features_train_PCA.shape

        self.classifier = SLogisticRegression()
        thetas = np.array(self.pca.S[:self.pca.k_components])
        thetas = thetas[:, None]
        print ' == theta dimensions : ', thetas.shape

        self.classifier.fit(thetas)

        # PCA for test data features
        print 'test features dimensions before PCA : ', features_test.shape
        features_test_PCA = self.pca.transform(features_test)
        print 'test features dimensions after PCA : ', features_test_PCA.shape

        # measure accuracy
        prediction = self.classifier.predict(features_test_PCA)
        self.accuracy = accuracy_score(prediction, labels_test)
        print 'accuracy : ', self.accuracy
        print '===-----------------------------------------------------------------------------==='
    #------------------------------------------------------------------------------------------------------------------

        # PCA for test data features
        #dataset =
        print 'test features dimensions before PCA : ', features_test.shape
        features_test_PCA = self.pca.transform(features_test)
        print 'test features dimensions after PCA : ', features_test_PCA.shape

        # measure accuracy
        prediction = self.classifier.predict(features_test_PCA)
        self.accuracy = accuracy_score(prediction, labels_test)
        print 'accuracy : ', self.accuracy
        print '===-----------------------------------------------------------------------------==='