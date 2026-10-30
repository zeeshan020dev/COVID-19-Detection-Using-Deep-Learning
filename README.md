# COVID-19 Detection Using Deep Learning

A deep learning-based chest X-ray classification project using **MobileNetV2 Transfer Learning** to classify images into four categories:

- COVID
- Lung Opacity
- Normal
- Viral Pneumonia

## 1. Project Overview and Objectives

This project explores the use of deep learning and transfer learning for classifying chest X-ray images.

The primary objective is to develop a convolutional neural network (CNN) using the pre-trained MobileNetV2 architecture to identify patterns in chest X-ray images and classify them into four categories.

### Objectives

- Apply transfer learning using MobileNetV2.
- Preprocess and classify chest X-ray images.
- Evaluate model performance using standard classification metrics.
- Understand the practical application of deep learning in medical image analysis.

## 2. Problem Statement

Analyzing chest X-ray images manually can be time-consuming and requires medical expertise.

This project investigates how a deep learning model can automatically classify chest X-ray images into four categories: COVID, Lung Opacity, Normal, and Viral Pneumonia.

The goal is to build an educational image-classification prototype and evaluate its performance on an unseen test dataset.

## 3. Dataset Information

The project uses the **COVID-19 Radiography Database** available on Kaggle.

**Dataset source:**  
[COVID-19 Radiography Database – Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

### Dataset Classes

| Class | Number of Images |
|---|---:|
| COVID | 3,616 |
| Lung Opacity | 6,012 |
| Normal | 10,192 |
| Viral Pneumonia | 1,345 |
| **Total** | **21,165** |

The dataset contains chest X-ray images organized into four categories.

**Note:** The dataset is not included in this repository. Download it from Kaggle and follow its licensing and attribution requirements.

For dataset setup instructions, see [`Dataset/README.md`](Dataset/README.md).

## 4. MobileNetV2 Architecture

This project uses **MobileNetV2**, a lightweight convolutional neural network architecture suitable for image classification tasks.

Transfer learning allows the model to reuse features learned from a large pre-trained image dataset and adapt them to the chest X-ray classification task.

### Model Workflow

1. Load and preprocess chest X-ray images.
2. Resize images to the model's required input dimensions.
3. Load the pre-trained MobileNetV2 base model.
4. Add a classification head for four output classes.
5. Train and validate the model.
6. Evaluate the trained model on unseen test images.

### Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 5. Model Evaluation Results

The trained model was evaluated on a separate test set containing **3,175 images**.

### Overall Performance

| Metric | Score |
|---|---:|
| Test Loss | 0.4606 |
| Test Accuracy | 82.83% |
| Weighted Precision | 84.14% |
| Weighted Recall | 82.83% |
| Weighted F1-Score | 83.05% |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| COVID | 0.70 | 0.85 | 0.77 |
| Lung Opacity | 0.86 | 0.79 | 0.83 |
| Normal | 0.90 | 0.82 | 0.86 |
| Viral Pneumonia | 0.67 | 0.98 | 0.79 |

The model achieved **82.83% test accuracy** across the four classes.

Performance varies between classes, so accuracy alone does not fully describe the model's classification capability.

## 6. Installation and Execution Instructions

### Prerequisites

- Python 3.9 or later
- pip
- Jupyter Notebook or Google Colab
- Kaggle account to access the dataset

### Step 1: Clone the Repository

```bash
git clone https://github.com/zeeshan020dev/COVID-19-Detection-Using-Deep-Learning.git

cd COVID-19-Detection-Using-Deep-Learning
````

### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

### Step 3: Download the Dataset

Download the dataset from:

[https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database?utm_source=chatgpt.com)

Extract it and configure the dataset path according to the notebook.

### Step 4: Run the Notebook

Open the notebook located in the `Code` directory:

`Code/covid-19-chest-x-ray-classification-using-dl.ipynb`

You can execute it using Jupyter Notebook or Google Colab.

### Step 5: Access the Trained Model

The trained model and class labels are available in the `Weights` directory.

- `covid_mobilenetv2.keras`
- `best_covid_mobilenetv2.keras`
- `class_labels.json`

## 7. Screenshots and Deployment Link

### Model Results

Add screenshots of the following outputs to demonstrate the model's performance:

- Training and validation accuracy
- Training and validation loss
- Confusion matrix
- Classification report
- Sample predictions

### Deployment

Live Demo: Not deployed yet.

Add your deployment URL here once the application is available online.

## 8. Medical-Use Disclaimer

This project is developed for educational and research purposes only.

It is an experimental deep learning classification model and has not been clinically validated.

It must not be used for medical diagnosis, treatment decisions, or patient care. Always consult qualified healthcare professionals for medical advice.

## Repository Structure

```
COVID-19-Detection-Using-Deep-Learning/
│
├── Code/
│   └── covid-19-chest-x-ray-classification-using-dl.ipynb
│
├── Dataset/
│   └── README.md
│
├── Weights/
│   ├── best_covid_mobilenetv2.keras
│   ├── class_labels.json
│   └── covid_mobilenetv2.keras
│
├── .gitattributes
├── LICENSE
├── README.md
└── requirements.txt
```

## Author

Muhammad Zeeshan Islam

\- GitHub: [@zeeshan020dev](https://github.com/zeeshan020dev?utm_source=chatgpt.com)

If you find this project useful, consider giving the repository a star.
