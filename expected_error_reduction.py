import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, precision_recall_fscore_support
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer

import time

# Load and split data
data = load_breast_cancer()
X, y = data.data, data.target

X_train_initial, X_pool, y_train_initial, y_pool = train_test_split(
    X, y, test_size=0.8, stratify=y, random_state=42
)

X_pool, X_test, y_pool, y_test = train_test_split(
    X_pool, y_pool, test_size=0.2, stratify=y_pool, random_state=42
)

# Base model
base_model = LogisticRegression(max_iter=1000, solver='lbfgs')

# Train initial model
model = clone(base_model)
model.fit(X_train_initial, y_train_initial)

# Performance evaluation
def evaluate_metrics(model, X, y):
    y_pred = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    return {
        'accuracy': model.score(X, y),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

performance_history = [model.score(X_test, y_test)]
metrics_history = [evaluate_metrics(model, X_test, y_test)]

# --- Expected Error Reduction strategy ---
def expected_error_reduction_manual(model, X_pool, X_train, y_train, possible_labels, validation_set):
    errors = []

    for i, x in enumerate(X_pool):
        x = x.reshape(1, -1)
        probs = model.predict_proba(x).flatten()

        expected_error = 0

        for label in possible_labels:
            # Simulate teaching this sample with this label
            new_X = np.vstack([X_train, x])
            new_y = np.hstack([y_train, label])

            temp_model = clone(model)
            temp_model.fit(new_X, new_y)

            # Estimate error on validation set (e.g., log loss)
            y_val_pred = temp_model.predict_proba(validation_set)
            val_error = log_loss(y_pool, y_val_pred, labels=possible_labels)

            # Weight by probability of label
            expected_error += probs[label] * val_error

        errors.append(expected_error)

    # Return index of sample with lowest expected error
    return np.argmin(errors)

# Active learning loop
n_queries = 10
for i in range(n_queries):
    print(f"\nQuery {i+1}/{n_queries}...", flush=True)
    start = time.time()

    # Query strategy: EER
    query_idx = expected_error_reduction_manual(
        model, X_pool, X_train_initial, y_train_initial,
        possible_labels=np.unique(y),
        validation_set=X_pool  # or use X_test if preferred
    )

    query_instance = X_pool[query_idx].reshape(1, -1)
    query_label = y_pool[query_idx].reshape(1,)

    # Teach the model
    X_train_initial = np.vstack([X_train_initial, query_instance])
    y_train_initial = np.hstack([y_train_initial, query_label])

    # Remove from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

    # Retrain model
    model = clone(base_model)
    model.fit(X_train_initial, y_train_initial)

    # Record performance
    current_metrics = evaluate_metrics(model, X_test, y_test)
    metrics_history.append(current_metrics)
    performance_history.append(current_metrics['accuracy'])

    print(f"Done in {time.time() - start:.2f}s | Accuracy: {current_metrics['accuracy']:.3f}")

# Final results
print("\nFinal Results:")
print(f"Initial Accuracy: {performance_history[0]:.3f}")
print(f"Final Accuracy: {performance_history[-1]:.3f}")
print(f"Final Precision: {metrics_history[-1]['precision']:.3f}")
print(f"Final Recall: {metrics_history[-1]['recall']:.3f}")
print(f"Final F1: {metrics_history[-1]['f1']:.3f}")

# Plotting
plt.plot(range(1, len(performance_history)), performance_history[1:], label='Expected Error Reduction', marker='o')
plt.title('Expected Error Reduction using Breast Cancer dataset')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.xlim((1, len(performance_history)-1))
plt.xticks(range(1, len(performance_history), 2))
plt.ylim((0, 1.2))
plt.grid(True)
plt.legend()
plt.show()

# Plot all metrics: accuracy, precision, recall, f1
plt.figure(figsize=(12, 6))
metrics_names = ['accuracy', 'precision', 'recall', 'f1']

for metric in metrics_names:
    values = [m[metric] for m in metrics_history]
    plt.plot(range(len(values)), values, label=metric.capitalize(), marker='o')

plt.xlabel('Number of Queries')
plt.ylabel('Score')
plt.title('Performance Metrics Over Time (EER Strategy)')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.05)
plt.xticks(range(len(metrics_history)))
plt.show()
