# Sport vs Politics Document Classification using Machine Learning

## Overview
This project implements a document-level text classifier that classifies a given news document as either:
- Politics
- Sports

The system uses classic Machine Learning techniques and text feature representations such as:
- Bag of Words (BoW)
- TF-IDF
- n-grams (unigrams + bigrams)

We compare three different ML algorithms:
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
The project is part of an academic assignment focused on text classification and model comparison.

## Dataset
Source: Kaggle (https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset?resource=download)

Format: CSV file with two columns:
- text → Paragraph-length news document
- label → Class label
    - 0 = Politics
    - 1 = Sports

Each row represents a full text document, not just a short sentence or headline.
The dataset is relatively clean and well-labeled, which results in very high classification accuracy.

## Features Used
- Bag of Words (BoW): Word count based representation
- TF-IDF: Term Frequency–Inverse Document Frequency weighting
- n-grams: Unigrams and bigrams to capture short phrases like “prime minister” or “world cup”

Vocabulary size is limited using max_features=5000 to:
- Reduce noise
- Control memory usage
- Improve generalization

## Machine Learning Models
We trained and evaluated the following models:
- Multinomial Naive Bayes (with BoW features)
- Logistic Regression (with TF-IDF + n-grams)
- Support Vector Machine (Linear SVM) (with TF-IDF + n-grams)

## Evaluation Metrics
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

## Results (Summary)
| Model               | Features           | Accuracy |
| ------------------- | ------------------ | -------- |
| Naive Bayes         | BoW                | ~0.996   |
| Logistic Regression | TF-IDF (1–2 grams) | ~1.000   |
| SVM                 | TF-IDF (1–2 grams) | ~1.000   |

A shuffled-label sanity check was also performed, which resulted in ~50% accuracy, confirming that the high performance is not due to data leakage.

## Report
A detailed 5+ page project report is included/submitted separately, covering:
- Data collection and description
- Preprocessing and feature extraction
- Model descriptions
- Experimental setup
- Quantitative comparison
- Error analysis
- Limitations

## Limitations
- Only two classes: Sports and Politics
- Dataset is relatively clean and easy, leading to very high accuracy
- Models are shallow (linear) and do not capture deep semantics

