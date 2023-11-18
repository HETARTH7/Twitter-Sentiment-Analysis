# Suicide Intent Detection

## Overview

This repository contains code for a machine learning project focused on suicide intent detection. The project involves preparing a dataset, performing data preprocessing, and implementing a text classification model to identify instances of suicidal intent in text data.

## Table of Contents

- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Classification Model](#classification-model)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Dataset

The dataset used in this project is a combination of various sources, including happy moments, emotional tweets, and suicidal data. The data is preprocessed and organized to create a balanced dataset for training and testing the classification model.

## Data Preparation

The code includes data preparation steps such as hyperlink removal, removal of non-ASCII characters, stop words removal, tokenization, stemming, and POS tagging. These steps are essential for creating a clean and meaningful corpus for training the machine learning model.

## Classification Model

A machine learning classification model is implemented using scikit-learn. The model is trained to predict whether a given text instance contains suicidal intent or not. Feature extraction techniques, including TF-IDF vectorization, tokenized text, and POS tags, are used to represent the text data.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/suicide-intent-detection.git
   cd suicide-intent-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:

   ```bash
   python main.py
   ```

4. Evaluate the model's performance and make predictions on new data.

## Dependencies

- pandas
- numpy
- nltk
- scikit-learn

Install dependencies using:

```bash
pip install -r requirements.txt
```
