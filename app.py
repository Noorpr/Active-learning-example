import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from query_by_committee import query_active_learning
from uncertainty_sampling_s1 import uncertainty_sampling_func
from expected_error_reduction import active_learning_with_eer
from get_the_dataset import get_breast_cancer, get_iris
from sklearn.ensemble import RandomForestClassifier


st.title("Active Learning Strategies")
st.subheader("Here is a showdown for the three popular strategies that are used in Active Learning")
st.markdown("- Uncertainty Sampling")
st.markdown("- Query By Committee")
st.markdown("- Expected Error Reduction")


st.write("These strategies are used to select the most informative samples from the pool of unlabeled data")


# Datasets
breast_X, breast_y = get_breast_cancer()
iris_X, iris_y = get_iris()

st.sidebar.header("Options")
st.sidebar.write("choose what dataset you will compare the results on:")
option1 = st.sidebar.selectbox("Dataset:", ["Breast Cancer", "Iris"])
option2 = st.sidebar.selectbox("which metric do you want:", ['f1', 'recall', 'precision'])

# measures for qbc
performance_history_qbc = []
metrics_history_qbc = []


# measures for us
accuracy_list = []
metrics_list = []

# measures for eer
results = dict()
baseModel = RandomForestClassifier(random_state=42)

if st.sidebar.button("Apply Option"):
    if option1 == 'Breast Cancer':
        performance_history_qbc , metrics_history_qbc = query_active_learning(breast_X, breast_y)
        accuracy_list, metrics_list = uncertainty_sampling_func(breast_X, breast_y)
        results = active_learning_with_eer(baseModel, breast_X, breast_y,n_initial=10, n_queries=7)

    elif option1 == 'Iris':
        performance_history_qbc , metrics_history_qbc = query_active_learning(iris_X, iris_y)
        accuracy_list, metrics_list = uncertainty_sampling_func(iris_X, iris_y)
        results = active_learning_with_eer(baseModel, iris_X, iris_y,n_initial=10, n_queries=7)
    

    fig , ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
    ax[0].plot(performance_history_qbc, label='Accuracy', marker='o')
    ax[0].set_title('Query By Committee')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Accuracy')
    ax[0].grid(True)
    ax[0].legend()

    
    ax[1].plot(accuracy_list, label='Accuracy', marker='o')
    ax[1].set_title("Uncertainty Sampling")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(results['performance_history'], label='Accuracy', marker='o')
    ax[2].set_title("Expected Error Reduction")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Accuracy")
    ax[2].grid(True)
    ax[2].legend()

    fig.suptitle("all of three strategies")
    
    st.pyplot(fig)



    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot([m[option2] for m in metrics_list], label='Uncertainty Sampling', color='red')
    ax.plot([m[option2] for m in metrics_history_qbc], label='Query by Committee', color='green')
    metrics_history_eer = results['metrics_history']
    ax.plot([m[option2] for m in metrics_history_eer], label="Expected Error Reduction", color='orange')
    ax.set_xticks(range(0, 21, 2))
    ax.set_title(f"{option2} score comparison")
    ax.grid(True)

    st.pyplot(fig)


    st.write("Active Learning is used when Labeling data is expensive, time-consuming, or requires expert knowledge (like medical images, legal documents, etc.).")
    st.write("Passive Learning is used when Labeling cost is low or data is already labeled.")
# strategies







