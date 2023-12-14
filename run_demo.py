
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
import logging

import os
import zipfile
import warnings

from transformers import AutoTokenizer, pipeline
import torch
import argparse
import logging
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

device = torch.device('cuda')



tokenizer_1 = AutoTokenizer.from_pretrained('facebook/opt-1.3b')

# unlearning_model_path = '/scratch/sg7729/machine-unlearning/models/opt1.3b_unlearned'
original_model_path_1 = 'facebook/opt-1.3b'

# tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token = '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')
generator_1 = pipeline('text-generation', model=original_model_path_1,  tokenizer=tokenizer_1, device=device, max_length = 30)


def predict_1(prompt):
    completion = generator_1(prompt)[0]["generated_text"]
    return completion


tokenizer_2 = AutoTokenizer.from_pretrained('facebook/opt-2.7b')

# unlearning_model_path = '/scratch/sg7729/machine-unlearning/models/opt1.3b_unlearned'
original_model_path_2 = 'facebook/opt-2.7b'

# tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token = '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')
generator = pipeline('text-generation', model=original_model_path_2,  tokenizer=tokenizer_2, device=device, max_length = 30)

def predict_2(prompt):
    completion = generator_2(prompt)[0]["generated_text"]
    return completion

import numpy as np
import gradio as gr

demo = gr.Blocks()

def predict(prompt):
    completion = generator(prompt)[0]["generated_text"]
    return completion

with demo:
    gr.Markdown("Generate text using this demo.")
    with gr.Tabs():
        with gr.TabItem("Opt 1.3B Original"):
            with gr.Row():
                            
                text_input_1 = gr.Textbox()
                text_output_1 = gr.Textbox()
                demo_1 = gr.Interface(fn=predict_1, inputs=text_input_1, outputs=text_output_1)

        with gr.TabItem("Opt 2.7B Original"):
            with gr.Row():
                text_input_2 = gr.Textbox()
                text_output_2 = gr.Textbox()
            
                demo_2 = gr.Interface(fn=predict_2, inputs=text_input_2, outputs=text_output_2)
demo.launch(share=True)
