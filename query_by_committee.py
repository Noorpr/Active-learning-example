import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from modAL.models import ActiveLearner, Committee
from modAL.disagreement  import vote_entropy_sampling
from get_the_dataset import get_iris, get_breast_cancer


# Load the dataset
X, y = get_breast_cancer()

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

def evaluate_metrics(committee, X, y):
    y_pred = committee.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    return {
        'accuracy': committee.score(X, y),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Initialize performance metrics storage
performance_history = [committee.score(X_test, y_test)]
metrics_history = [evaluate_metrics(committee, X_test, y_test)]
n_queries = 20

# Active learning loop
for idx in range(n_queries):
    # Query for new instance
    query_idx, query_instance = committee.query(X_pool)
    
    # Get the true label for the queried instance
    queried_label = y_pool[query_idx]
    
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
    performance_history.append(current_metrics['accuracy'])

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(len(performance_history)), performance_history, marker='o', label='Accuracy')
plt.xlabel('Number of queries')
plt.ylabel('Accuracy')
plt.title('Active Learning Progress - Query by Committee')
plt.legend()
plt.grid(True)
plt.show()

# Print final results
print("\nFinal Results:")
print(f"Initial accuracy: {performance_history[0]:.3f}")
print(f"Final accuracy: {performance_history[-1]:.3f}")
print("\nFinal Metrics:")
print(f"Precision: {metrics_history[-1]['precision']:.3f}")
print(f"Recall: {metrics_history[-1]['recall']:.3f}")
print(f"F1 Score: {metrics_history[-1]['f1']:.3f}")

# Plot metrics history
plt.figure(figsize=(12, 6))
metrics_names = ['accuracy', 'precision', 'recall', 'f1']
for metric in metrics_names:
    plt.plot([m[metric] for m in metrics_history], label=metric.capitalize())
plt.xlabel('Number of queries')
plt.ylabel('Score')
plt.title('Performance Metrics Over Time')
plt.legend()
plt.grid(True)
plt.show()