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

import os
import zipfile
import warnings
import pandas as pd
from classifier import model as m
from classifier import data_preprocess

def tokenize_and_encode(tokenizer, comments, labels, max_length=128):
    # Initialize empty lists to store tokenized inputs and attention masks
    input_ids = []
    attention_masks = []

    # Iterate through each comment in the 'comments' list
    for comment in comments:
        # Tokenize and encode the comment using the BERT tokenizer
        encoded_dict = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Append the tokenized input and attention mask to their respective lists
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists of tokenized inputs and attention masks to PyTorch tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Convert the labels to a PyTorch tensor with the data type float32
    labels = torch.tensor(labels, dtype=torch.float32)

    # Return the tokenized inputs, attention masks, and labels as PyTorch tensors
    return input_ids, attention_masks, labels

def dataloader():
    # Set the path to the data
    data_path = '/scratch/sg7729/machine-unlearning/classifier/data/train.csv'

    # Set the path to the model
    model_path = '/scratch/sg7729/machine-unlearning/classifier/model/'

    # Set the path to the output directory
    output_path = '/scratch/sg7729/machine-unlearning/classifier/output/'

    # preprocess the data
    train = data_preprocess.preprocess(data_path)

    # Label names
    target_labels = [col for col in train.columns if train[col].dtypes == 'int64']
    # target_lables = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Get the cleaned comments and the target labels as lists
    comments = train['Cleaned_Comments'].to_list()
    data = train[target_labels]

    return train, comments, data, target_labels