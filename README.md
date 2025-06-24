# Summer School WnCC Submission

## Week 2 Assignment

This repository contains my Week 2 submission for the WnCC Summer School: **Machine Learning & Data Science** track.

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

### Run
1. Ensure `nltk` corpora are downloaded or available in `week2/nltk_data/`.
2. Ensure `GoogleNews-vectors-negative300.bin` is present in the root or modify path in `problem2.py`.
3. Run `problem1.py` and `problem2.py` in any Python environment or via terminal using:
```bash
python Problem\ 1/problem1.py
python Problem\ 2/problem2.py


