import numpy as np
import time
from sklearn.base import clone
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def evaluate_metrics(model, X, y):
    """Evaluate model performance with multiple metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted'),
        'recall': recall_score(y, y_pred, average='weighted'),
        'f1': f1_score(y, y_pred, average='weighted'),
        'log_loss': log_loss(y, y_prob)
    }

def expected_error_reduction(model, X_pool, X_train, y_train, validation_X, validation_y):
    """
    Expected Error Reduction strategy for active learning (sequential implementation).
    
    Parameters:
    -----------
    model : estimator
        The classifier model
    X_pool : array-like
        The unlabeled data pool
    X_train : array-like
        The labeled training data
    y_train : array-like
        The labels for training data
    validation_X : array-like
        Validation data features
    validation_y : array-like
        Validation data labels
        
    Returns:
    --------
    idx : int
        Index of the best instance to query
    """
    possible_labels = np.unique(y_train)
    probs = model.predict_proba(X_pool)
    expected_errors = {}
    
    # Calculate expected error for each instance in the pool
    for i, x in enumerate(X_pool):
        # Reshape single example to 2D
        x = x.reshape(1, -1)
        expected_error = 0
        
        # Try each possible label
        for label_idx, label in enumerate(possible_labels):
            # Create new training set with the labeled example
            new_X = np.vstack([X_train, x])
            new_y = np.hstack([y_train, [label]])
            
            # Train a new model
            temp_model = clone(model)
            temp_model.fit(new_X, new_y)
            
            # Calculate error on validation set
            val_probs = temp_model.predict_proba(validation_X)
            val_error = log_loss(validation_y, val_probs, labels=possible_labels)
            
            # Weight error by predicted probability of that label
            expected_error += probs[i][label_idx] * val_error
        
        expected_errors[i] = expected_error
    
    # Return index with lowest expected error
    return min(expected_errors.items(), key=lambda x: x[1])[0]

# Sample active learning loop
def active_learning_with_eer(base_model, X, y, n_initial=10, n_queries=20, test_size=0.3, random_state=42):
    """
    Perform active learning using Expected Error Reduction strategy.
    
    Parameters:
    -----------
    base_model : estimator
        Base classifier model
    X : array-like
        Full feature set
    y : array-like
        Full label set
    n_initial : int, default=10
        Number of initial training examples
    n_queries : int, default=20
        Number of queries to make
    test_size : float, default=0.3
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Results including performance history and training data
    """
    # Split into training pool and test set
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split validation set from pool (for EER evaluation)
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_pool, y_pool, test_size=0.2, random_state=random_state, stratify=y_pool
    )
    
    # Initialize with a small labeled training set
    indices = np.random.RandomState(random_state).choice(
        len(X_pool), size=n_initial, replace=False
    )
    X_train = X_pool[indices]
    y_train = y_pool[indices]
    
    # Remove initially selected examples from pool
    X_pool = np.delete(X_pool, indices, axis=0)
    y_pool = np.delete(y_pool, indices)
    
    # Performance tracking
    performance_history = []
    metrics_history = []
    
    # Initial model
    model = clone(base_model)
    model.fit(X_train, y_train)
    
    # Evaluate initial model
    initial_metrics = evaluate_metrics(model, X_test, y_test)
    metrics_history.append(initial_metrics)
    performance_history.append(initial_metrics['accuracy'])
    
    print(f"Initial accuracy: {initial_metrics['accuracy']:.3f}")
    
    # Active learning loop
    for i in range(n_queries):
        print(f"\nQuery {i+1}/{n_queries}...", flush=True)
        start = time.time()
        
        # Query using EER strategy
        query_idx = expected_error_reduction(
            model, X_pool, X_train, y_train, X_val, y_val
        )
        
        # Add queried instance to training data
        query_instance = X_pool[query_idx].reshape(1, -1)
        query_label = y_pool[query_idx].reshape(1,)
        
        X_train = np.vstack([X_train, query_instance])
        y_train = np.append(y_train, query_label)
        
        # Remove from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
        
        # Retrain model
        model = clone(base_model)
        model.fit(X_train, y_train)
        
        # Evaluate and record performance
        current_metrics = evaluate_metrics(model, X_test, y_test)
        metrics_history.append(current_metrics)
        performance_history.append(current_metrics['accuracy'])
        
        print(f"Done in {time.time() - start:.2f}s | Accuracy: {current_metrics['accuracy']:.3f}")
    
    return {
        'model': model,
        'X_train': X_train,
        'y_train': y_train,
        'performance_history': performance_history,
        'metrics_history': metrics_history
    }

# Example usage:
# from sklearn.ensemble import RandomForestClassifier
# base_model = RandomForestClassifier(n_estimators=100, random_state=42)
# results = active_learning_with_eer(base_model, X, y, n_initial=10, n_queries=20)
# 
# # Plot learning curve
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(results['performance_history'])), results['performance_history'])
# plt.xlabel('Number of queries')
# plt.ylabel('Test accuracy')
# plt.title('Active Learning with Expected Error Reduction')
# plt.grid(True)
# plt.show()