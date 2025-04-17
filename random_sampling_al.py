import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Callable, Dict, List, Tuple, Any, Optional
import pandas as pd
from get_the_dataset import get_breast_cancer, get_iris


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to evaluate
    X_test : array-like
        Test features
    y_test : array-like
        True test labels
        
    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics

def random_sampling_strategy(unlabeled_indices: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Select samples randomly from the unlabeled pool
    
    Parameters:
    -----------
    unlabeled_indices : array-like
        Indices of unlabeled samples
    batch_size : int
        Number of samples to select
        
    Returns:
    --------
    query_indices : array-like
        Indices of selected samples
    """
    return np.random.choice(unlabeled_indices, 
                           min(batch_size, len(unlabeled_indices)), 
                           replace=False)

def active_learning_iteration(
    model: Any,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_pool: np.ndarray,
    y_pool_true: np.ndarray,
    unlabeled_indices: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Perform one iteration of active learning
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to train
    X_labeled, y_labeled : array-like
        Currently labeled data
    X_pool, y_pool_true : array-like
        Pool of unlabeled data and their true labels
    unlabeled_indices : array-like
        Indices of samples in the unlabeled pool
    X_test, y_test : array-like
        Test data for evaluation
    sampling_strategy : Callable
        Function to select samples from unlabeled pool
    batch_size : int
        Number of samples to select
        
    Returns:
    --------
    model : sklearn estimator
        Trained model
    X_labeled, y_labeled : array-like
        Updated labeled data
    unlabeled_indices : array-like
        Updated unlabeled indices
    metrics : Dict[str, float]
        Performance metrics
    """
    # Train model on currently labeled data
    model.fit(X_labeled, y_labeled)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Select samples using the provided strategy
    query_indices = random_sampling_strategy(unlabeled_indices, batch_size)
    
    # Get labels for the selected samples (simulating oracle/human labeling)
    X_queried = X_pool[query_indices]
    y_queried = y_pool_true[query_indices]
    
    # Add newly labeled samples to the labeled dataset
    X_labeled_new = np.vstack((X_labeled, X_queried))
    y_labeled_new = np.concatenate((y_labeled, y_queried))
    
    # Remove queried samples from unlabeled pool
    unlabeled_indices_new = np.setdiff1d(unlabeled_indices, query_indices)
    
    return model, X_labeled_new, y_labeled_new, unlabeled_indices_new, metrics

def run_random_sampling(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 10,
    max_iterations: int = 10
) -> List[Dict[str, float]]:
    """
    Run the complete active learning process
    
    Parameters:
    -----------
    X_initial, y_initial : array-like
        Initial labeled data
    X_pool, y_pool_true : array-like
        Pool of unlabeled data and their true labels
    X_test, y_test : array-like
        Test data for evaluation
    sampling_strategy : Callable
        Function to select samples from unlabeled pool
    batch_size : int
        Number of samples to select in each iteration
    max_iterations : int
        Maximum number of active learning iterations
        
    Returns:
    --------
    metrics_history : List[Dict[str, float]]
        History of performance metrics
    """
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initially label only a small portion of the training data
    X_initial, X_pool, y_initial, y_pool = train_test_split(X_train, y_train, 
                                                           test_size=0.9, random_state=42)
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Initialize labeled and unlabeled data
    X_labeled = X_initial.copy()
    y_labeled = y_initial.copy()
    
    unlabeled_indices = np.arange(len(X_pool))
    metrics_history = []
    
    for iteration in range(max_iterations):
        # Perform one iteration of active learning
        model, X_labeled, y_labeled, unlabeled_indices, metrics = active_learning_iteration(
            model, X_labeled, y_labeled, X_pool, y_pool, unlabeled_indices,
            X_test, y_test, batch_size
        )
        
        # Store metrics
        metrics["iteration"] = iteration
        metrics["labeled_samples"] = len(X_labeled)
        metrics_history.append(metrics)
        
        # Print progress
        print(f"Iteration {iteration}, Metrics: {metrics}, Labeled samples: {len(X_labeled)}")
        
        # Check if we've exhausted the unlabeled pool
        if len(unlabeled_indices) <= batch_size:
            print("No more unlabeled samples available")
            break
    
    return metrics_history

def plot_learning_curves(metrics_history: List[Dict[str, float]]) -> None:
    """
    Plot learning curves for multiple metrics
    
    Parameters:
    -----------
    metrics_history : List[Dict[str, float]]
        History of performance metrics
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(metrics_history)
    
    # Plot all metrics except iteration and labeled_samples
    metrics_to_plot = [col for col in df.columns if col not in ['iteration', 'labeled_samples']]
    
    plt.figure(figsize=(12, 8))
    for metric in metrics_to_plot:
        plt.plot(df['iteration'], df[metric], marker='o', label=metric)
    
    plt.xlabel('Active Learning Iteration')
    plt.ylabel('Score')
    plt.title('Active Learning Performance Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage with a synthetic dataset
if __name__ == "__main__":

    pass