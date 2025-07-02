from transformers import TrainingArguments
print(TrainingArguments.__module__)

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from sklearn.metrics import accuracy_score, f1_score

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch_device}")

imdb_raw_dataset = load_dataset("imdb")

bert_model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_sequence_classifier_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2).to(torch_device)

def imdb_tokenize_example(example):
    return bert_tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

imdb_tokenized_dataset = imdb_raw_dataset.map(imdb_tokenize_example, batched=True)
imdb_tokenized_dataset = imdb_tokenized_dataset.remove_columns(["text"])
imdb_tokenized_dataset.set_format("torch")

def imdb_compute_metrics(pred):
    predicted_labels = np.argmax(pred.predictions, axis=1)
    true_labels = pred.label_ids
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return {"accuracy": accuracy, "f1": f1}

imdb_training_args = TrainingArguments(
    output_dir="./bert_imdb_results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

imdb_trainer = Trainer(
    model=bert_sequence_classifier_model,
    args=imdb_training_args,
    train_dataset=imdb_tokenized_dataset["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=imdb_tokenized_dataset["test"].select(range(1000)),
    compute_metrics=imdb_compute_metrics
)

imdb_trainer.train()

imdb_eval_results = imdb_trainer.evaluate()
print("Evaluation Results:", imdb_eval_results)

imdb_model_save_path = "./bert_finetuned_imdb"
bert_sequence_classifier_model.save_pretrained(imdb_model_save_path)
bert_tokenizer.save_pretrained(imdb_model_save_path)
print(f"Model saved to {imdb_model_save_path}")

print("\n--- Sample Inference ---")
imdb_loaded_model = AutoModelForSequenceClassification.from_pretrained(imdb_model_save_path).to(torch_device)
imdb_loaded_tokenizer = AutoTokenizer.from_pretrained(imdb_model_save_path)
imdb_sentiment_pipeline = pipeline("sentiment-analysis", model=imdb_loaded_model, tokenizer=imdb_loaded_tokenizer, device=0 if torch.cuda.is_available() else -1)

imdb_sample_review_text = "I absolutely loved the movie! It was fantastic."
print(f"Text: {imdb_sample_review_text}")
print("Prediction:", imdb_sentiment_pipeline(imdb_sample_review_text))
