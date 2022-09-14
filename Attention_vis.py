import pandas as pd
import numpy as np
import sys
import yaml

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm

from transformers import AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from rdkit import Chem

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from packaging import version

import torchmetrics
from torchmetrics import R2Score

from PolymerSmilesTokenization import PolymerSmilesTokenizer
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


class DownstreamRegression(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        self.PretrainedModel = deepcopy(PretrainedModel)
        self.PretrainedModel.resize_token_embeddings(len(tokenizer))

        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.Regressor(logits)
        return output


def main(attention_config):
    if attention_config['task'] == 'pretrain':
        smiles = attention_config['smiles']
    else:
        data = pd.read_csv(attention_config['file_path'])
        smiles = data.values[attention_config['index'],0]

    if attention_config['add_vocab_flag']:
        vocab_sup = pd.read_csv(attention_config['vocab_sup_file'], header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    encoding = tokenizer(
        str(smiles),
        add_special_tokens=True,
        max_length=attention_config['blocksize'],
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    if attention_config['task'] == 'pretrain':
        outputs = PretrainedModel(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    else:
        model = DownstreamRegression(drop_rate=0).to(device)
        checkpoint = torch.load(attention_config['model_path'])
        model.load_state_dict(checkpoint['model'])
        model = model.double()

        model.eval()
        with torch.no_grad():
            outputs = model.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

    attention = outputs[-1]

    fig, axes = plt.subplots(3,4, figsize=(attention_config['figsize_x'],attention_config['figsize_y']))
    xticklabels = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

    if attention_config['task'] == 'pretrain':

        for i in range(3):
            for j in range(4):
                sns.heatmap(attention[attention_config['layer']][0,4*i+j,:,:].cpu().detach().numpy(), ax = axes[i,j], xticklabels=xticklabels, yticklabels=xticklabels)
                axes[i,j].set_title(label="Attention Head %s" % str(4*i+j+1), fontsize=attention_config['fontsize'])
                axes[i,j].tick_params(labelsize=attention_config['labelsize'])
                cbar = axes[i,j].collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=attention_config['labelsize'])

    else:
        yticklabels = ['Layer 1','Layer 2','Layer 3','Layer 4','Layer 5','Layer 6']
        for i in range(3):
            for j in range(4):
                for layer in range(6):
                    attention_sub = attention[layer][0,4*i+j,0,:].cpu().detach().numpy().reshape(1,-1)
                    if layer == 0:
                        attention_CLS = attention_sub
                    else:
                        attention_CLS = np.vstack((attention_CLS, attention_sub))
                sns.heatmap(attention_CLS, ax = axes[i,j], xticklabels=False)
                axes[i, j].set_title(label="Attention Head %s" % str(4 * i + j + 1), fontsize=attention_config['fontsize'])
                axes[i, j].set_yticklabels(rotation=attention_config['rotation'], labels=yticklabels)
                axes[i, j].tick_params(labelsize=attention_config['labelsize'])
                cbar = axes[i, j].collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=attention_config['labelsize'])
    plt.savefig(attention_config['save_path'], bbox_inches='tight')

if __name__ == "__main__":

    attention_config = yaml.load(open("config_attention.yaml", "r"), Loader=yaml.FullLoader)

    """Device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PretrainedModel = RobertaModel.from_pretrained(attention_config['pretrain_path']).to(device)
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=attention_config['blocksize'])

    main(attention_config)

