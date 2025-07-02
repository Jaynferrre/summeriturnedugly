# Week 3: Sentiment Analysis with BERT using Hugging Face Transformers

This project implements a machine learning pipeline to perform sentiment analysis on the IMDb dataset using Hugging Face's `transformers` and `datasets` libraries. The model is a fine-tuned `bert-base-uncased` transformer for binary classification (positive or negative review).

## Pipeline Components

- **Dataset Loading**: IMDb dataset from the Hugging Face `datasets` library.
- **Preprocessing**: Tokenization using the BERT tokenizer with padding, truncation, and max length 512.
- **Model Training**: Fine-tuning `bert-base-uncased` using a custom PyTorch training loop. This replaced the Hugging Face `Trainer` API due to compatibility errors.
- **Evaluation**: Model performance was evaluated using accuracy and F1-score metrics on a test subset.
- **Inference**: Final sentiment predictions were generated using Hugging Face’s `pipeline` for demonstration.

## Performance

The model was trained on a 2000-sample subset of the IMDb training split for two epochs and evaluated on 1000 test samples. Loss values decreased steadily, and evaluation metrics indicated effective learning even on a limited sample.

## Challenges and Mitigation

We initially attempted to fine-tune the model using Hugging Face's `Trainer` and `TrainingArguments`. However, repeated `TypeError` exceptions related to valid arguments like `evaluation_strategy` arose despite using the correct library versions. After extensive debugging and environment checks, we resolved this by replacing the Trainer API with a fully custom PyTorch training loop, which allowed us full control and compatibility.

Additionally, training was initially slow on CPU, leading to high system load. GPU compatibility was verified using `torch.cuda.is_available()` and `nvidia-smi`, enabling future scaling to accelerated environments.
