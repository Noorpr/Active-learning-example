import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling # this will change according to what you choose from strategies
from get_the_dataset import get_iris, get_breast_cancer # we need to change the dataset because the accuracy is actually 100% for both
from random_state_generator import generate
import warnings
warnings.filterwarnings('ignore')


RANDOM_STATE = generate(42)

X, y = get_breast_cancer()

X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

iterations = 10
accuracy_list = []


learner = ActiveLearner(
    estimator = RandomForestClassifier(n_estimators=100, random_state= RANDOM_STATE),
    query_strategy = uncertainty_sampling,
    X_training = X_train,
    y_training = y_train
)

for i in range(iterations):
    query_idx, query_instance = learner.query(X_pool)
    learner.teach(X_pool[query_idx], y_pool[query_idx])

    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

    accuracy_list.append(accuracy_score(y_train, learner.predict(X_train)))


print(accuracy_list)
