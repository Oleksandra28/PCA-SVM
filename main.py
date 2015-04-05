__author__ = 'osopova'


from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_olivetti_faces

from sklearn.cross_validation import  train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from SPCA import SPCA

from ClassificationManager import ClassificationManager
from SLogisticRegression import SLogisticRegression

#########################################################################################################
if __name__ == "__main__":

    dataset = fetch_lfw_people(min_faces_per_person=50 )
    datamanager = ClassificationManager(dataset)


    # dataset = fetch_olivetti_faces()
    # datamanager = ClassificationManager(dataset, classifier)
