from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Value
import torch
import numpy as np
import evaluate
from transformers import EarlyStoppingCallback
from datasets import load_dataset, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import set_seed
import random


#creating a directory to store the results for the current run
current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
dir_name = f"run_{current_time}"
#dir_name = "exp_37"
dir_name = os.path.join("results", dir_name)
os.makedirs(dir_name, exist_ok=True)


#custom_seed = random.randint(0, 2**32 - 1) # randomizing seed
custom_seed = 50 #keeping the same seed for comparisons
random.seed(custom_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(custom_seed)
set_seed(custom_seed)
with open(os.path.join(dir_name, "seed.txt"), "w") as f:
    f.write(str(custom_seed))


#Model and tokenizer
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

#enabling cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

#getting data from the dataset and its preparation
dataset = load_dataset("google/code_x_glue_cc_code_refinement", "small")

small_train = dataset["train"].shuffle(seed=42).select(range(800))
small_eval  = dataset["validation"].shuffle(seed=42).select(range(100))
small_test  = dataset["test"].shuffle(seed=42).select(range(100))

def filter_long_examples(example):
    max_len = 256
    return (
        len(tokenizer.tokenize(example["buggy"])) <= max_len
        and len(tokenizer.tokenize(example["fixed"])) <= max_len
    )

# filter before mapping
print("sizes")
print(small_train)
print(small_eval)
print(small_test)
small_train = small_train.filter(filter_long_examples)
small_eval = small_eval.filter(filter_long_examples)
small_test = small_test.filter(filter_long_examples)

print("sizes after removal")
print(small_train)
print(small_eval)
print(small_test)

def duplicate_bug_fix(dataset_split):
    buggy_texts = list(dataset_split["buggy"])
    fixed_texts = list(dataset_split["fixed"])
    
    all_texts  = buggy_texts + fixed_texts
    all_labels = [1]*len(buggy_texts) + [0]*len(fixed_texts)
    
    new_dataset = Dataset.from_dict({"code": all_texts, "labels": all_labels})

    new_dataset = new_dataset.shuffle(seed=42)
    
    return new_dataset

train_dataset = duplicate_bug_fix(small_train)
eval_dataset  = duplicate_bug_fix(small_eval)
test_dataset  = duplicate_bug_fix(small_test)

def preprocess(examples):
    preprocessed_data = tokenizer(examples["code"], truncation=True, padding="max_length", max_length=256)
    preprocessed_data["labels"] = examples["labels"]
    return preprocessed_data

train_dataset = train_dataset.map(preprocess, batched=True, load_from_cache_file=False)
eval_dataset  = eval_dataset.map(preprocess,  batched=True, load_from_cache_file=False)
test_dataset  = test_dataset.map(preprocess,  batched=True, load_from_cache_file=False)

train_dataset = train_dataset.cast_column("labels", Value("int64"))
eval_dataset  = eval_dataset.cast_column("labels", Value("int64"))
test_dataset  = test_dataset.cast_column("labels", Value("int64"))

cols = ["input_ids", "attention_mask", "labels"]
train_dataset.set_format(type="torch", columns=cols)
eval_dataset.set_format(type="torch", columns=cols)
test_dataset.set_format(type="torch", columns=cols)


#Setting metrics
metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

#Training args and trainer definition
training_args = TrainingArguments(
    output_dir=dir_name,
    optim = "adamw_torch", #changing for different runs - adafactor/adamw_torch
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.03,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=512,
    save_total_limit = 2,
    seed=custom_seed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

# Training
trainer.train()


#FINAL CHECKS
#check for training set
train_results = trainer.evaluate(train_dataset)
print("Train results:", train_results)

# check for validation set
eval_results = trainer.evaluate(eval_dataset)
print("Validation results:", eval_results)

# check for test set
test_output = trainer.predict(test_dataset)
test_metrics = test_output.metrics
print("Test results:", test_metrics)


#SAVING INFORMATION
# Saving results to txt files
#validation results for the final model
with open(os.path.join(dir_name,"training_results.txt"), "w", encoding="utf-8") as f:
    f.write(str(train_results))
#validation results for the final model
with open(os.path.join(dir_name,"validation_results.txt"), "w", encoding="utf-8") as f:
    f.write(str(eval_results))
#test results for the final model
with open(os.path.join(dir_name,"test_results.txt"), "w", encoding="utf-8") as f:
    f.write(str(test_metrics))

print("Saved validation and test results separately.")

#everything below is commented for short experiment runs

#saving model and tokenizer
best_model_dir = os.path.join(dir_name, "best_model")
os.makedirs(best_model_dir, exist_ok=True)

model.save_pretrained(best_model_dir)
tokenizer.save_pretrained(best_model_dir)

# Saving current model info
with open(os.path.join(best_model_dir, "current_model_info.txt"), "w", encoding="utf-8") as f:
    f.write(str(trainer.model))

#Saving best model checkpoint info
with open(os.path.join(best_model_dir,"best_model_checkpoint.txt"), "w") as f:
    f.write(trainer.state.best_model_checkpoint)

# Saving logs
df = pd.DataFrame(trainer.state.log_history)
df.to_csv(os.path.join(dir_name,"training_log.csv"), index=False)


#SAVING PLOTS
# Training loss 
train_df = df.dropna(subset=["loss"])
plt.figure(figsize=(8,5))
plt.plot(train_df["step"], train_df["loss"], marker="o")
plt.xlabel("Iteracja")
plt.ylabel("Funkcja straty")
plt.grid()
plt.savefig(os.path.join(dir_name,"training_loss.png"))
plt.close()

# Evaluation loss
eval_df = df.dropna(subset=["eval_loss"])
plt.figure(figsize=(8,5))
plt.plot(eval_df["epoch"], eval_df["loss"], marker="o")
plt.xlabel("Epoka")
plt.ylabel("Funkcja straty")
plt.grid()
plt.savefig(os.path.join(dir_name,"eval_loss.png"))
plt.close()


# Learning rate
lr_df = df.dropna(subset=["learning_rate"])
plt.figure(figsize=(8,5))
plt.plot(lr_df["step"], lr_df["learning_rate"], marker="x", color="orange")
plt.xlabel("Iteracja")
plt.ylabel("Współczynnik uczenia")
plt.grid()
plt.savefig(os.path.join(dir_name,"learning_rate.png"))
plt.close()

# Validation accuracy
eval_df = df.dropna(subset=["eval_accuracy"])
plt.figure(figsize=(8,5))
plt.plot(eval_df["epoch"], eval_df["eval_accuracy"], marker="s", color="green")
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.grid()
plt.savefig(os.path.join(dir_name,"validation_accuracy.png"))
plt.close()



#PLOTTING CONFUSION MATRIXES

# Confusion Matrix - training set
train_output = trainer.predict(train_dataset)
y_true_train = train_output.label_ids
y_pred_train = np.argmax(train_output.predictions, axis=-1)
cm_train = confusion_matrix(y_true_train, y_pred_train, labels=[0,1])


disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["0", "1"]) #0 - fixed - no errors, 1 - buggy - errors
disp_train.plot(cmap="Oranges")
plt.title("Macierz pomyłek dla zbioru treningowego")
plt.savefig(os.path.join(dir_name,"confusion_matrix_train.png"))
plt.close()


# Confusion Matrix - validation set
val_output = trainer.predict(eval_dataset)
y_true_val = val_output.label_ids
y_pred_val = np.argmax(val_output.predictions, axis=-1)

cm_val = confusion_matrix(y_true_val, y_pred_val, labels=[0,1])


disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=["0", "1"]) #0 - fixed - no errors, 1 - buggy - errors
disp_val.plot(cmap="Blues")
plt.title("Macierz pomyłek dla zbioru walidacyjnego")
plt.savefig(os.path.join(dir_name,"confusion_matrix_val.png"))
plt.close()


# Confusion Matrix - test set
y_true_test = test_output.label_ids
y_pred_test = np.argmax(test_output.predictions, axis=-1)
cm_test = confusion_matrix(y_true_test, y_pred_test, labels=[0,1])

disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["0", "1"]) #0 - fixed - no errors, 1 - buggy - errors
disp_test.plot(cmap="Greens")
plt.title("Macierz pomyłek dla zbioru testowego")
plt.savefig(os.path.join(dir_name,"confusion_matrix_test.png"))
plt.close()

print("Code finished successfully, everything should be saved.")