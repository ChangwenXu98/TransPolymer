from transformers import (RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer,
    TrainingArguments)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import sys
import os
import yaml

"""Import PolymerSmilesTokenizer from PolymerSmilesTokenization.py"""
from PolymerSmilesTokenization import PolymerSmilesTokenizer

"""Import LoadPretrainData"""
from dataset import LoadPretrainData

"""Device"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available()) #checking if CUDA+GPU works

"""train-validation split"""
def split(file_path):
    dataset = pd.read_csv(file_path, header=None).values
    train_data, valid_data = train_test_split(dataset, test_size=0.2, random_state=1)
    return train_data, valid_data

def main(pretrain_config):
    """Use Roberta configuration"""
    config = RobertaConfig(
        vocab_size=50265,
        max_position_embeddings=pretrain_config['max_position_embeddings'],
        num_attention_heads=pretrain_config['num_attention_heads'],
        num_hidden_layers=pretrain_config['num_hidden_layers'],
        type_vocab_size=1,
        hidden_dropout_prob=pretrain_config['hidden_dropout_prob'],
        attention_probs_dropout_prob=pretrain_config['attention_probs_dropout_prob'],
    )

    """Load Data"""
    train_data, valid_data = split(pretrain_config['file_path'])
    print(train_data.shape, valid_data.shape, valid_data)

if __name__ == "__main__":
    pretrain_config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    """Run the main function"""
    main(pretrain_config)