import pandas as pd
import numpy as np
import sys
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn import MaxPool1d

from transformers import AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from openTSNE import TSNE

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from packaging import version

import torchmetrics
from torchmetrics import R2Score

from PolymerSmilesTokenization import PolymerSmilesTokenizer
from dataset import Dataset_Emb, TransPolymerEmbeddings

def emb_convert(file_path, tokenizer, config):
    data = pd.read_csv(file_path)
    dataset = Dataset_Emb(data, tokenizer, tsne_config['blocksize'], config)
    dataloader = DataLoader(dataset, tsne_config['batch_size'], shuffle=False, num_workers=0)
    for step, batch in tqdm(enumerate(dataloader)):
        batch = batch.to(device)
        embeddings = batch.squeeze()
        embeddings = torch.transpose(embeddings, dim0=1, dim1=2)
        max_pool = MaxPool1d(kernel_size=tsne_config['blocksize'], padding=0,
                             dilation=1)  # Apply max pooling for conversion into t-SNE input
        embeddings = max_pool(embeddings)
        embeddings = torch.transpose(embeddings, dim0=1, dim1=2).reshape(embeddings.shape[0],
                                                                         768).cpu().detach().numpy()
        if step == 0:
            print("shape of embedding:", embeddings.shape)
            embeddings_all = embeddings
        else:
            embeddings_all = np.vstack((embeddings_all, embeddings))

    return  embeddings_all

def main(tsne_config):

    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=tsne_config['blocksize'])
    config = RobertaConfig(
            vocab_size=50265,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

    pretrain_data = emb_convert(tsne_config['pretrain_path'], tokenizer, config)
    PE_I_data = emb_convert(tsne_config['PE_I_path'], tokenizer, config)
    PE_II_data = emb_convert(tsne_config['PE_II_path'], tokenizer, config)
    Egc_data = emb_convert(tsne_config['Egc_path'], tokenizer, config)
    Egb_data = emb_convert(tsne_config['Egb_path'], tokenizer, config)
    Eea_data = emb_convert(tsne_config['Eea_path'], tokenizer, config)
    Ei_data = emb_convert(tsne_config['Ei_path'], tokenizer, config)
    Xc_data = emb_convert(tsne_config['Xc_path'], tokenizer, config)
    EPS_data = emb_convert(tsne_config['EPS_path'], tokenizer, config)
    Nc_data = emb_convert(tsne_config['Nc_path'], tokenizer, config)
    OPV_data = emb_convert(tsne_config['OPV_path'], tokenizer, config)

    print("start fitting t-SNE")
    tSNE = TSNE(
        perplexity=tsne_config['perplexity'],
        metric=tsne_config['metric'],
        n_jobs=tsne_config['n_jobs'],
        verbose=tsne_config['verbose'],
    )
    pretrain_tSNE = tSNE.fit(pretrain_data)
    print("finish fitting")
    PE_I_tSNE = pretrain_tSNE.transform(PE_I_data)
    PE_II_tSNE = pretrain_tSNE.transform(PE_II_data)
    Egc_tSNE = pretrain_tSNE.transform(Egc_data)
    Egb_tSNE = pretrain_tSNE.transform(Egb_data)
    Eea_tSNE = pretrain_tSNE.transform(Eea_data)
    Ei_tSNE = pretrain_tSNE.transform(Ei_data)
    Xc_tSNE = pretrain_tSNE.transform(Xc_data)
    EPS_tSNE = pretrain_tSNE.transform(EPS_data)
    Nc_tSNE = pretrain_tSNE.transform(Nc_data)
    OPV_tSNE = pretrain_tSNE.transform(OPV_data)
    print("finish t-SNE")

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(pretrain_tSNE[:, 0], pretrain_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='lightgrey',
               label=tsne_config["pretrain_label"])
    ax.scatter(PE_I_tSNE[:, 0], PE_I_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='maroon', label=tsne_config["PE-I_label"])
    ax.scatter(PE_II_tSNE[:, 0], PE_II_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='coral', label=tsne_config["PE-II_label"])
    ax.scatter(Egc_tSNE[:, 0], Egc_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='darkorange', label=tsne_config["Egc_label"])
    ax.scatter(Egb_tSNE[:, 0], Egb_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='gold', label=tsne_config["Egb_label"])
    ax.scatter(Eea_tSNE[:, 0], Eea_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='lawngreen', label=tsne_config["Eea_label"])
    ax.scatter(Ei_tSNE[:, 0], Ei_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='green', label=tsne_config["Ei_label"])
    ax.scatter(Xc_tSNE[:, 0], Xc_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='cyan', label=tsne_config["Xc_label"])
    ax.scatter(EPS_tSNE[:, 0], EPS_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='blue', label=tsne_config["EPS_label"])
    ax.scatter(Nc_tSNE[:, 0], Nc_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='violet', label=tsne_config["Nc_label"])
    ax.scatter(OPV_tSNE[:, 0], OPV_tSNE[:, 1], s=50, edgecolors='None', linewidths=0.4, c='deeppink', label=tsne_config["OPV_label"])

    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis('off')
    ax.legend(fontsize=20, loc='upper left')
    plt.savefig(tsne_config['save_path'], bbox_inches='tight')


if __name__ == "__main__":

    tsne_config = yaml.load(open("config_tSNE.yaml", "r"), Loader=yaml.FullLoader)

    """Device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    main(tsne_config)
