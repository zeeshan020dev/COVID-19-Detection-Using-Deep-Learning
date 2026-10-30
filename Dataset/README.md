# Dataset Information

## Overview

This project uses the **COVID-19 Radiography Database**, a publicly available chest X-ray image dataset hosted on Kaggle.

The dataset contains chest X-ray images categorized into four classes. It is used to train and evaluate a deep learning model for multi-class image classification.

**Dataset Source:**  
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

## Dataset Classes

The dataset contains the following four classes:

| Class | Number of Images |
|---|---:|
| COVID | 3,616 |
| Lung_Opacity | 6,012 |
| Normal | 10,192 |
| Viral Pneumonia | 1,345 |
| **Total** | **21,165** |

## Download Instructions

The dataset is not included in this repository because of its size and licensing conditions.

To download it:

1. Visit the [COVID-19 Radiography Database on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).
2. Sign in to your Kaggle account.
3. Accept the dataset's terms and conditions, if required.
4. Click **Download** to download the dataset.
5. Extract the downloaded archive.
6. Update the dataset path in the Jupyter Notebook to match your local or Kaggle environment.

## Dataset Folder Structure

After downloading and extracting the dataset, its directory structure should look similar to this:

```text
COVID-19_Radiography_Dataset/
│
├── COVID/
│   └── images/
│
├── Lung_Opacity/
│   └── images/
│
├── Normal/
│   └── images/
│
└── Viral Pneumonia/
    └── images/
````

Each class folder contains an `images` directory with the corresponding chest X-ray images.

## Source Attribution and Licensing

Original Dataset: COVID-19 Radiography Database\
Source: Kaggle\
Dataset Link: [https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database?utm_source=chatgpt.com)

The dataset is provided by its original contributors. Please refer to the Kaggle dataset page and its accompanying documentation for the applicable licence, attribution requirements, and usage restrictions.

This repository's MIT licence applies to the original project code only. It does not override the dataset's licence or grant additional rights to the dataset images.

Please credit the original dataset contributors when using this dataset in academic or research work.

**Important:** The image counts above reflect the dataset version used in this project. Dataset contents or counts may change in future releases.
