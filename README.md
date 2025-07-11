<<<<<<< HEAD
# Summer School WnCC Submission

## Week 2 Assignment

This repository contains my Week 2 submission for WnCC's Introduction to Machine Learning and NLP Summer School

###  Problem 1 — SMS Spam Classification
- **Model**: Logistic Regression
- **Data**: `spam.csv`
- **Accuracy Achieved**: ~94%

###  Problem 2 — Airline Tweet Sentiment Analysis
- **Model**: Logistic Regression + Word2Vec
- **Data**: `Tweets.csv`
- **Accuracy Achieved**: ~76%

### Dependencies
- Python 3.9+
- `nltk`, `gensim`, `scikit-learn`, `pandas`, `numpy`, `contractions`


# Week 3: Sentiment Analysis with BERT using Hugging Face Transformers

This project implements a machine learning pipeline to perform sentiment analysis on the IMDb dataset using Hugging Face's `transformers` and `datasets` libraries. The model is a fine-tuned `bert-base-uncased` transformer for binary classification (positive or negative review).

## Pipeline Components

- **Dataset Loading**: IMDb dataset from the Hugging Face `datasets` library.
- **Preprocessing**: Tokenization using the BERT tokenizer with padding, truncation, and max length 512.
- **Model Training**: Fine-tuning `bert-base-uncased` using a custom PyTorch training loop. This replaced the Hugging Face `Trainer` API due to compatibility errors.
- **Evaluation**: Model performance was evaluated using accuracy and F1-score metrics on a test subset.
- **Inference**: Final sentiment predictions were generated using Hugging Face’s `pipeline` for demonstration.
=======
# Transformer-based Language Model for Next-Word Prediction

This project implements a transformer-based language model using GPT-2, fine-tuned on a subset of the WikiText-2 dataset. The objective is to predict the next word in a sequence, a foundational task in natural language processing (NLP) with applications in text generation, autocomplete systems, and writing assistants.

## Objectives

- Fine-tune a pre-trained GPT-2 model for next-word prediction.
- Apply best practices for tokenizer alignment, data preprocessing, and model training.
- Evaluate model performance using perplexity and top-k accuracy.

## Dataset

- **WikiText-2**: A clean subset of Wikipedia articles used for structured language learning.
- Loaded using the `datasets` library from Hugging Face.

## Model

- **Base Model**: `gpt2` from Hugging Face Transformers.
- Fine-tuned using the `Trainer` API with `GPT2LMHeadModel`.

## Evaluation Metrics

- **Perplexity**: Measures the uncertainty of the model in predicting the next word.
- **Top-k Accuracy**: Percentage of times the correct next word was among the top-k predictions (k=5 in this case).
>>>>>>> ad023c5 (Initial commit: Fine-tuned GPT-2 for next-word prediction)

