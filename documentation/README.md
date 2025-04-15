# 🎓 Faculty of Computer and Artificial Intelligence – Cairo University

## 📘 Assignment: Active Learning Strategies

---

👤 **Name:** Noor Elden Tariq Mohammed Medhat  
📅 **Academic Year:** 2024–2025

---

## 📑 Table of Contents

1. [📖 Introduction](#-introduction)  
2. [⚖️ Active vs. Passive Learning](#-active-vs-passive-learning)  
3. [📊 Datasets](#-datasets)  
4. [🧠 Strategies](#-strategies)  
5. [🔬 Experiments (Methodology)](#-experiments-methodology)  
6. [📈 Results](#-results)  
7. [✅ Conclusion](#-conclusion)  
8. [🔗 References](#-references)

---

## 📖 Introduction

Active learning is a machine learning technique where the model chooses the data it learns from. Instead of training on a fully labeled dataset, the model selects the most **informative or uncertain samples** to be labeled by a human (oracle).  
This reduces labeling costs while still achieving **high accuracy**.

📌 Common in:  
- 🖼️ Image classification  
- 📚 NLP  
- 🏥 Medical diagnosis

---

## ⚖️ Active vs Passive Learning

| 🔍 Active Learning                         | 📥 Passive Learning                        |
|-------------------------------------------|--------------------------------------------|
| Chooses what data to learn from           | Uses all labeled data                      |
| Focuses on uncertain or informative cases | No selection or prioritization             |
| More efficient with fewer labeled samples | May require more labeled data              |

---

## 📊 Datasets

### 🩺 Breast Cancer Dataset
- Features from digitized images of **breast masses**
- Classes: **Malignant** or **Benign**
- ⚠️ Imbalanced dataset → needs careful handling

### 🌸 Iris Dataset
- Classic dataset with **150 flower samples**
- 3 Species: Setosa, Versicolor, Virginica  
- Balanced: 50 samples per class  
- 4 features per sample:  
  - Sepal Length, Sepal Width  
  - Petal Length, Petal Width

---

## 🧠 Strategies

### 🔻 Uncertainty Sampling
- Picks samples where prediction confidence is **lowest**
- E.g., predicted probability close to 0.5  
- Boosts learning by targeting confusion points

### 🧑‍⚖️ Query by Committee (QBC)
- Trains multiple models (a **committee**)  
- Selects samples with **most disagreement**  
- More diverse insights → better decision boundary

### 🎯 Expected Error Sampling
- Predicts which sample would most reduce future model error  
- Selects based on **expected improvement** in accuracy

---

## 🔬 Experiments (Methodology)

### Initial Training
- Train models on a small initial labeled dataset.
- Establish baseline performance metrics.

### Active Selection
- Apply different strategies to select informative samples.
- Request labels for the selected data points.

### Model Updating
- Retrain models with newly labeled data.
- Evaluate performance improvements after each iteration.

---

## 📈 Results

### Query by Committee (QBC)

![QBC - Breast Cancer](https://github.com/Noorpr/Active-learning-example/blob/main/QBC-BreastCancer.png)

![QBC - IRIS][https://github.com/Noorpr/Active-learning-example/blob/main/QBC-IRIS.png]

1. First Plot:
- Starts around 0.90 with fluctuations.
- Peaks above 0.94 after the 10th query.
- Shows variability but overall improvement.

2. Second Plot:
- Begins around 0.92, dips, then stabilizes at 0.96.
- Achieves consistent high accuracy after initial queries.
