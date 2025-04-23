# Consumer-Complaint-Sentiment-Analysis
AIT 526 - Team Project
Team Members: Yasser Jaghoori, Andrej Paskalov, Yaseen Trombati
Date: March 26, 2025

## Project Overview
This project performs sentiment analysis on consumer complaint narratives to classify the emotional tone of complaints submitted to financial service companies. Using natural language processing (NLP) techniques and machine learning models, we aim to identify patterns in customer dissatisfaction.

## Dataset
The analysis is based on a cleaned dataset of consumer complaints extracted from a structured Excel file:
complaints_cleaned.xlsx

## Key fields used include:

Complaint ID

Consumer complaint narrative

Issue, Product, State, etc.

## Methodology
The pipeline includes the following stages:

Data Preprocessing

Loading data with pandas

Text cleaning using re, NLTK, and tokenization

Stopword removal and lemmatization

Negative word tagging using the opinion_lexicon from NLTK

NLP & Text Analytics

Named Entity Recognition using spaCy

Sentiment analysis via TextBlob and NaiveBayesAnalyzer

Frequency distribution of words in complaints

Machine Learning

Encoding categorical features with LabelEncoder

## Classification using:

K-Nearest Neighbors (KNN)

Decision Tree Classifier

Naive Bayes Classifier

Model evaluation via classification_report

## Visualization

Tree visualizations with matplotlib

Word distributions and frequency plots

## Tools & Libraries
Python 3

Pandas, NumPy

spaCy (en_core_web_sm)

NLTK (opinion_lexicon, punkt)

TextBlob

scikit-learn

matplotlib

## Running the Notebook
Make sure the following requirements are installed:

pip install pandas numpy matplotlib spacy nltk textblob scikit-learn openpyxl
python -m spacy download en_core_web_sm
