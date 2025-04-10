from sklearn.datasets import load_iris, load_breast_cancer

def get_iris():
    iris = load_iris()
    X, y = iris.data, iris.target

    return X, y

def get_breast_cancer():
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target

    return X, y
