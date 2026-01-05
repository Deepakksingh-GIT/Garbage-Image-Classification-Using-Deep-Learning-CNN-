# Hi there, I'm Deepak Kumar Singh 👋
## Enthusiastic Data Science Student

# ♻️ RecycleVision – Garbage Image Classification Using Deep Learning

## 📌 Project Overview

RecycleVision is a **deep learning–based computer vision project** that classifies garbage images into different waste categories such as **plastic, metal, glass, paper, cardboard, and trash**. The goal of this project is to support **automated waste segregation** and improve recycling efficiency using **Convolutional Neural Networks (CNNs) and Transfer Learning**.

The trained model is deployed using a **Streamlit web application**, allowing users to upload an image of waste and instantly receive a prediction with confidence score.

---

## 🗑️ Problem Statement

The objective of this project is to build a deep learning–based image classification system that can accurately categorize waste images into classes such as plastic, metal, glass, paper, cardboard, and organic/trash. This solution aims to support automated waste segregation by identifying garbage types from images using a trained deep learning model, deployed through a simple and user-friendly interface.

---

## 💼 Business Use Cases

### ♻️ Smart Recycling Bins

Automatically identify and sort waste into the correct recycling bins, reducing human effort and improving recycling efficiency.

### 🏙️ Municipal Waste Management

Reduce manual waste sorting time, labor costs, and operational inefficiencies in large-scale waste processing systems.

### 📚 Educational & Awareness Tools

Help individuals and institutions learn proper waste segregation practices through visual and interactive tools.

### 📊 Environmental Analytics

Analyze waste composition and recycling trends to support data-driven environmental and sustainability initiatives.

---

## 📂 Dataset

**Dataset Used:** Garbage Classification (6 Classes) – Kaggle

**Classes:**

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

**Total Images:** ~2,467

---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow, Keras
* **Model Type:** CNN, Transfer Learning (MobileNetV2)
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
* **Web App:** Streamlit

---

## 🔄 Project Workflow

1. Data Collection
2. Data Preprocessing & Augmentation
3. Exploratory Data Analysis (EDA)
4. Model Building using Transfer Learning
5. Model Training & Evaluation
6. Model Selection
7. Streamlit App Development
8. Deployment

---

## 🧹 Data Preprocessing

* Image resizing to **224 × 224**
* Pixel normalization (0–1 range)
* Data augmentation:

  * Rotation
  * Zoom
  * Horizontal flipping
* Handling class imbalance using augmentation

---

## 🧠 Model Architecture

* Pretrained **MobileNetV2** (ImageNet weights)
* Frozen base layers
* Custom fully connected layers
* Softmax output layer for multi-class classification

**Why MobileNetV2?**

* Lightweight and fast
* High accuracy
* Suitable for deployment

---

## 📈 Model Evaluation

**Metrics Used:**

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

**Best Model Performance:**

* Accuracy: **~88%**
* Balanced precision and recall
* Minimal class-wise misclassification

---

## 🖥️ Streamlit Web Application

**Features:**

* Upload garbage image
* Predict waste category
* Display confidence score
* Simple and intuitive UI

**Run App Locally:**

```bash
streamlit run app.py
```

---

## 📊 Results

* Successfully classified waste images into correct categories
* Achieved accuracy greater than **85%**
* Fast inference suitable for real-time use

---

## 📚 Key Learnings

* Image preprocessing and augmentation
* CNN and transfer learning concepts
* Model evaluation techniques
* Deploying deep learning models using Streamlit


