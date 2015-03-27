__author__ = 'osopova'


from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_olivetti_faces

from sklearn.cross_validation import  train_test_split
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from SPCA import SPCA





#########################################################################################################
if __name__ == "__main__":

    dataset = fetch_lfw_people(min_faces_per_person=50 )
    data = dataset.data

    print dataset.images.shape
    # introspect the images arrays to find the shapes (for plotting)
    samples, h, w = dataset.images.shape

    print 'shape of dataset.data ', data.shape
    features = data.shape[1]
    print 'features ', features
    labels = dataset.target

    features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.25)

    principalComponents = len(features_train)*0.15
    pca = SPCA(principalComponents)
    pca.fit(features_train)
    features_train_PCA = pca.transform(features_train)
    features_test_PCA = pca.transform(features_test)
    print 'features_train_PCA.shape', features_train_PCA.shape

    classifier = SVC(C = 1000, cache_size=200, class_weight='auto', coef0=0, degree=3,gamma = 0.005, kernel = 'rbf', probability=False, random_state=None)
    classifier.fit(features_train_PCA, labels_train)
    prediction = classifier.predict(features_test_PCA)
    print accuracy_score(prediction, labels_test)
    print '=========================================================================================================='


    dataset = fetch_olivetti_faces()
    data = dataset.data
    print 'shape of dataset.data ', data.shape
    features = data.shape[1]
    print 'features ', features
    labels = dataset.target

    features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.25)

    principalComponents = len(features_train)*0.15
    pca = SPCA(principalComponents)
    pca.fit(features_train)
    features_train_PCA = pca.transform(features_train)
    features_test_PCA = pca.transform(features_test)
    print 'features_train_PCA.shape', features_train_PCA.shape

    classifier = SVC(C = 1000, cache_size=200, class_weight='auto', coef0=0, degree=3,gamma = 0.005, kernel = 'rbf', probability=False, random_state=None)
    classifier.fit(features_train_PCA, labels_train)
    prediction = classifier.predict(features_test_PCA)
    print accuracy_score(prediction, labels_test)
    print '=========================================================================================================='















