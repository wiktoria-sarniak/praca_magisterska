from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch
import numpy as np
import evaluate
from transformers import EarlyStoppingCallback
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from transformers import set_seed
import random


#creating a directory to store the results for the current run
current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
dir_name = f"run_{current_time}"
#dir_name = "exp_118"

dir_name = os.path.join("results_codebert_fix", dir_name)
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
model_name = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#enabling cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

#getting data from the dataset and its preparation
dataset = load_dataset("google/code_x_glue_cc_code_refinement", "small")

train_dataset = dataset["train"].shuffle(seed=42) #.select(range(800))
eval_dataset  = dataset["validation"].shuffle(seed=42) #.select(range(100))
test_dataset  = dataset["test"].shuffle(seed=42) #.select(range(100))

def filter_long_examples(example):
    max_len = 256
    return (
        len(tokenizer.tokenize(example["buggy"])) <= max_len
        and len(tokenizer.tokenize(example["fixed"])) <= max_len
    )

# filter before mapping
train_data = train_dataset.filter(filter_long_examples)
eval_data = eval_dataset.filter(filter_long_examples)
test_data = test_dataset.filter(filter_long_examples)


def preprocess(batch):
    #inputs = tokenizer(batch["buggy"], text_target=batch["fixed"], max_length=256, truncation=True, padding="max_length")
    #return inputs
    model_inputs = tokenizer(
        batch["buggy"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        batch["fixed"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    labels_ids = np.array([
        [(token if token != tokenizer.pad_token_id else -100) for token in label] 
        for label in labels["input_ids"]
    ])
    model_inputs["labels"] = labels_ids

    return model_inputs


train_dataset = train_dataset.map(preprocess, batched=True, load_from_cache_file=False)
eval_dataset  = eval_dataset.map(preprocess,  batched=True, load_from_cache_file=False)
test_dataset  = test_dataset.map(preprocess,  batched=True, load_from_cache_file=False)

train_dataset.set_format(type="torch")
eval_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")

print(train_dataset[0])

#Setting metrics
metric_bleu = evaluate.load("bleu")
metric_rouge = evaluate.load("rouge")
metric_chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    #needed to avoid Overflow error
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    references = [[label] for label in decoded_labels]
    preds = [pred for pred in decoded_preds]          
    
    
    return {
        "bleu" : metric_bleu.compute(predictions=preds, references=references)["bleu"],
        "rouge" : metric_rouge.compute(predictions=preds, references=references),
        "chrf" : metric_chrf.compute(predictions=preds, references=references), 
        "rougeLSum" : metric_rouge.compute(predictions=preds, references=references)["rougeLsum"]
    }


args = Seq2SeqTrainingArguments(
    #
    learning_rate=5e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    lr_scheduler_type="linear",
    #
    output_dir=dir_name,
    eval_strategy="epoch",
    optim = "adamw_torch",
    save_strategy="epoch",
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    save_total_limit=2,
    num_train_epochs=50,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rougeLSum",
    predict_with_generate=True,
    logging_dir=os.path.join(dir_name, "logs"),
    logging_steps=100,
    seed=custom_seed
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

#Training
trainer.train()

df = pd.DataFrame(trainer.state.log_history)


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

# Validation bleu
eval_df = df.dropna(subset=["eval_bleu"])
plt.figure(figsize=(8,5))
plt.plot(eval_df["step"], eval_df["eval_bleu"], marker="s", color="green")
plt.xlabel("Iteracja")
plt.ylabel("Metryka bleu")
plt.grid()
plt.savefig(os.path.join(dir_name,"validation_bleu.png"))
plt.close()

# Validation rougeLsum
eval_df = df.dropna(subset=["eval_rougeLSum"])
plt.figure(figsize=(8,5))
plt.plot(eval_df["step"], eval_df["eval_rougeLSum"], marker="s", color="green")
plt.xlabel("Iteracja")
plt.ylabel("Metryka bleu")
plt.grid()
plt.savefig(os.path.join(dir_name,"validation_rougeLsum.png"))
plt.close()

print("Code finished successfully, everything should be saved.")