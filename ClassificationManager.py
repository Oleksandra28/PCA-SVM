__author__ = 'osopova'

from SPCA import SPCA

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class ClassificationManager():

    def __init__(self, dataset, classifier, components_percent = 0.15, crossvalidation_percent = 0.2, test_percent = 0.2):

        self.components_percent = components_percent
        self.crossvalidation_percent = crossvalidation_percent
        self.test_percent = test_percent
        self.classifier = classifier

        self.data_set = dataset
        self.data = dataset.data

        self.pca = None

    #------------------------------------------------------------------------------------------------------------------

        features = self.data.shape[1]
        print 'initial features dimensions : ', features
        labels = dataset.target

        features_train, features_test, labels_train, labels_test = train_test_split(self.data, labels, test_size = self.test_percent)

        # TODO move this to classifier fit
        # save crossvalidation part of the dataset
        # features_train_len = len(features_train)
        # labels_train_len = len(labels_train)
        #
        # features_crossvalidation = features_train[:features_train_len*self.crossvalidation_percent]
        # labels_crossvalidation = labels_train[:labels_train_len*self.crossvalidation_percent]
        # features_train = features_train - features_crossvalidation
        # labels_train = labels_crossvalidation - labels_crossvalidation

        self.pca =  SPCA(self.components_percent)
        self.pca.fit(features_train)

        print 'train features dimensions before PCA : ', features_train.shape
        features_train_PCA = self.pca.transform(features_train)

        print 'train features dimensions after PCA : ', features_train_PCA.shape
        self.classifier.fit(features_train_PCA, labels_train)

        print 'test features dimensions before PCA : ', features_test.shape
        features_test_PCA = self.pca.transform(features_test)
        print 'test features dimensions after PCA : ', features_test_PCA.shape

        self.classifier.fit(features_train_PCA, labels_train)
        prediction = self.classifier.predict(features_test_PCA)
        self.accuracy = accuracy_score(prediction, labels_test)
        print 'accuracy : ', self.accuracy
        print '========================================================================================================'
    #------------------------------------------------------------------------------------------------------------------