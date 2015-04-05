__author__ = 'osopova'


from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_olivetti_faces

from sklearn.cross_validation import  train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from SPCA import SPCA


from ClassificationManager import ClassificationManager

from sklearn import linear_model

from SLogisticRegression import SLogisticRegression

#########################################################################################################
if __name__ == "__main__":

    dataset = fetch_lfw_people(min_faces_per_person=50 )
    #classifier = SVC(C = 1000, cache_size=200, class_weight='auto', coef0=0, degree=3,gamma = 0.005, kernel = 'rbf', probability=False, random_state=None)

    #classifier = linear_model.LogisticRegression()
    classifier = SLogisticRegression()
    datamanager = ClassificationManager(dataset, classifier, components_percent = 0.99)


    #
    # dataset = fetch_olivetti_faces()
    # datamanager = ClassificationManager(dataset, classifier)
    #
    # classifier = GaussianNB()
    # datamanager = ClassificationManager(dataset, classifier)

    # print '=================================================================================='
    #
    # classifier = linear_model.LogisticRegression()
    # datamanager = ClassificationManager(dataset, classifier)
    #
    # print '=================================================================================='
    #
    # classifier = SLogisticRegression()
    # datamanager = ClassificationManager(dataset, classifier)