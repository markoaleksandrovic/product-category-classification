
# Automatic Product Category Classification

## Project Overview

This project develops a Machine Learning system that
automatically predicts product categories based on
product titles.

The goal is to improve product classification in
e-commerce platforms by reducing manual work and
increasing classification accuracy.

## Dataset

The dataset contains 35,311 original products.

After data cleaning and preprocessing:
- Products: 30,762
- Categories: 10
- Training samples: 24,609
- Test samples: 6,153

Dataset file: `products.csv`

## Data Preprocessing

The following preprocessing steps were applied:

1. Removal of missing product titles and categories.
2. Text normalization.
3. Removal of duplicate product titles.
4. Removal of conflicting category labels.
5. Standardization of category names.

Examples of category standardization:

- CPU -> CPUs
- Mobile Phone -> Mobile Phones
- fridge -> Fridges

## Feature Engineering

Two TF-IDF approaches were tested:

- Word-level TF-IDF (unigrams and bigrams)
- Character-level TF-IDF (3-5 character n-grams)

Character-level TF-IDF provided better results,
especially for product titles containing technical
model numbers.

## Machine Learning Models

Three approaches were evaluated:

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Multinomial Naive Bayes | 93.09% | 0.9309 |
| Logistic Regression (Word TF-IDF) | 94.91% | 0.9512 |
| Logistic Regression (Character TF-IDF) | 97.94% | 0.9796 |

## Final Model

The selected model uses:

- TF-IDF Vectorizer
- Character n-grams (3, 5)
- Logistic Regression
- Scikit-learn Pipeline

The model achieved 97.94% accuracy on the
existing test dataset.

This is the best observed experimental result;
additional independent validation would be
needed to confirm generalization performance.

The trained model is saved as:

`models/product_classifier.pkl`

## Project Structure

    product-category-classification/
    |
    |-- products.csv
    |-- train_model.py
    |-- predict_category.py
    |-- requirements.txt
    |-- README.md
    |
    |-- models/
    |   |-- product_classifier.pkl
    |
    |-- notebooks/
        |-- 01_product_classification_EDA.ipynb

## Installation

Clone the repository:

    git clone https://github.com/YOUR_USERNAME/product-category-classification.git

Navigate to the project folder:

    cd product-category-classification

Install dependencies:

    python -m pip install -r requirements.txt


## Training the Model

Run:

    python train_model.py

This script:
- Loads the dataset
- Cleans and preprocesses product titles
- Standardizes category labels
- Trains the ML pipeline
- Evaluates model performance
- Saves the trained model

## Predicting Product Categories

Run:

    python predict_category.py

Enter a product title when prompted.

Example:

    Enter product title: apple iphone 7 32gb
    Predicted category: Mobile Phones

Type `exit` to close the application.

## Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The best experimental model achieved:

- Accuracy: 97.94%
- Macro F1: 0.9796

## Limitations

The model may incorrectly classify products with
short or ambiguous titles.

For example, some product model numbers do not
contain enough information to clearly identify
the correct category.

Future improvements may include:
- Additional training data
- Hyperparameter tuning
- Confidence-based manual review
- Independent validation
- Improved feature engineering

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab
- GitHub

## Author

Machine Learning course project.

## License

Educational project. Dataset redistribution is
subject to the original data provider's terms.
