import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
import re
from wordcloud import WordCloud, STOPWORDS
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import os
import zipfile
import warnings

import model as m
import data_preprocess as data_preprocess
from utils import tokenize_and_encode, dataloader

warnings.filterwarnings('ignore')

# Set the device to GPU
device = torch.device('cuda')
model = m.get_model()
model.to(device)

#set tokenizer
tokenizer = m.get_tokenizer()

# Set the random seed for reproducible results
random_seed = 42

# load data from utils.py
train, comments, data, target_labels = dataloader()

# Split the data into training and validation sets
Train_texts, Test_texts, Train_labels, Test_labels = train_test_split(
    comments, train[target_labels].values, test_size=0.2, random_state=2023)

#validation set
test_texts, val_texts, test_labels, val_labels = train_test_split(
    Test_texts, Test_labels, test_size=0.5, random_state=23)

print('numbers of training Dataset ',len(Train_texts))
print('numbers of testing Dataset',len(test_texts))
print('numbers of validation Dataset',len(val_texts))


# Tokenize and Encode the comments and labels for the training set
input_ids, attention_masks, labels = tokenize_and_encode(
    tokenizer, 
    Train_texts, 
    Train_labels
)

# Step 4: Tokenize and Encode the comments and labels for the test set
test_input_ids, test_attention_masks, test_labels = tokenize_and_encode(
    tokenizer,
    test_texts,
    test_labels
)

# Tokenize and Encode the comments and labels for the validation set
val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(
    tokenizer,
    val_texts,
    val_labels
)

# Creating DataLoader for the balanced dataset
batch_size = 32
train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#test
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#val
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print('Batch Size :',train_loader.batch_size)
Batch =next(iter(train_loader))
print('Each Input ids shape :',Batch[0].shape)
print('Input ids :\n',Batch[0][0])
print('Corresponding Decoded text:\n',tokenizer.decode(Batch[0][0]))
print('Corresponding Attention Mask :\n',Batch[1][0])
print('Corresponding Label:',Batch[2][0])

# Optimizer setup
optimizer = AdamW(model.parameters(), lr=0.00005)

train_loss_history = []
train_acc_history=[]
val_loss_history = []
val_accuracy_history = []

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs):
    model=model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions=0
        total_predictions=0
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [t.to(device) for t in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            train_loss_history.append(loss.item())
            logits = outputs.logits
            predictions = torch.round(torch.sigmoid(logits))
            correct_predictions += torch.sum(predictions == labels).item()
            total_predictions += labels.size(0)*labels.size(1)
            if (step + 1) % 150 == 0:
                print(f'Epoch {epoch+1}, Step {step+1}, Training Loss: {loss.item()}')
            if(step+1)==780:
                break
        train_accuracy=correct_predictions/total_predictions
        train_acc_history.append(train_accuracy)
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [t.to(device) for t in batch]

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits

                predictions = torch.round(torch.sigmoid(logits))
                correct_predictions += torch.sum(predictions == labels).item()
                total_predictions += labels.size(0)*labels.size(1)
                
        val_accuracy = correct_predictions / total_predictions

        val_loss_history.append(val_loss / len(val_loader))
        val_accuracy_history.append(val_accuracy)

        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)},Train Accuracy: {train_accuracy}, Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}')

train_model(model, train_loader, val_loader, optimizer, device, num_epochs=15)


# Save the tokenizer and model in the same directory
output_dir = "/models/bert-finetuned"
model.save_pretrained(output_dir)  # Save model's state dictionary and configuration
tokenizer.save_pretrained(output_dir)  # Save tokenizer's configuration and vocabulary