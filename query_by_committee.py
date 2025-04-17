import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from modAL.models import ActiveLearner, Committee
from modAL.disagreement  import vote_entropy_sampling

def evaluate_metrics(committee, X, y):
    y_pred = committee.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    return {
        'accuracy': committee.score(X, y),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def query_active_learning(X, y):
    # Split the data into training, validation, and test sets
    X_train_initial, X_pool, y_train_initial, y_pool = train_test_split(
        X, y, test_size=0.8, random_state=42, stratify=y
    )

    X_pool, X_test, y_pool, y_test = train_test_split(
        X_pool, y_pool, test_size=0.2, random_state=42, stratify=y_pool
    )

    # Initialize committee members
    n_committee = 3
    committee_members = [
        ActiveLearner(
            estimator = RandomForestClassifier(n_estimators=100, random_state= i),
            X_training = X_train_initial,
            y_training = y_train_initial
        )
        for i in range(n_committee)
    ]

    # Initialize the Committee
    committee = Committee(
        learner_list=committee_members,
        query_strategy=vote_entropy_sampling
    )

    # Initialize performance metrics storage
    metrics_history = [evaluate_metrics(committee, X_test, y_test)]
    n_queries = 20

    # Active learning loop
    for idx in range(n_queries):
        # Query for new instance
        query_idx, _ = committee.query(X_pool)
        
        # Teach the committee
        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1,)
        )
        
        # Remove the queried instance from the pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
        
        # Calculate and store performance metrics
        current_metrics = evaluate_metrics(committee, X_test, y_test)
        metrics_history.append(current_metrics)
    return metrics_history
