from transformers import (RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer,
    TrainingArguments)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import sys
import os
import yaml
from rdkit import Chem
from copy import deepcopy
from packaging import version

"""Import PolymerSmilesTokenizer from PolymerSmilesTokenization.py"""
from PolymerSmilesTokenization import PolymerSmilesTokenizer

"""Pretrain Dataset"""
class LoadPretrainData(Dataset):
    def __init__(self, tokenizer, dataset, blocksize):
        self.tokenizer = tokenizer
        self.blocksize = blocksize
        self.dataset = dataset

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, i):
        smiles = self.dataset[i][0]

        encoding = self.tokenizer(
            str(smiles),
            add_special_tokens=True,
            max_length=self.blocksize,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
        )

"""Downstream Dataset"""
class Downstream_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_token_len = max_token_len

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, i):
        data_row = self.dataset.iloc[i]
        seq = data_row[0]
        prop = data_row[1]

        encoding = self.tokenizer(
            str(seq),
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            prop=prop
        )

"""Adapted from RobertaEmbedding"""
class TransPolymerEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

class Dataset_Emb(Dataset):
    def __init__(self, dataset, tokenizer, max_token_len, config):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_token_len = max_token_len
        self.Embeddings = TransPolymerEmbeddings(config)

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, i):
        data_row = self.dataset.iloc[i]
        smiles = data_row[0]

        encoding = self.tokenizer(
            str(smiles),
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return self.Embeddings(input_ids=encoding["input_ids"])

class DataAugmentation:
    def __init__(self, aug_indicator):
        super(DataAugmentation, self).__init__()
        self.aug_indicator = aug_indicator

    """Rotate atoms to generate more SMILES"""
    def rotate_atoms(self, li, x):
        return (li[x % len(li):] + li[:x % len(li)])

    """Generate SMILES"""
    def generate_smiles(self, smiles):
        smiles_list = []
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            mol = None
        if mol != None:
            n_atoms = mol.GetNumAtoms()
            n_atoms_list = [nat for nat in range(n_atoms)]
            if n_atoms != 0:
                for iatoms in range(n_atoms):
                    n_atoms_list_tmp = self.rotate_atoms(n_atoms_list, iatoms)  # rotate atoms' index
                    nmol = Chem.RenumberAtoms(mol, n_atoms_list_tmp)  # renumber atoms in mol
                    try:
                        smiles = Chem.MolToSmiles(nmol,
                                                  isomericSmiles=True,  # keep isomerism
                                                  kekuleSmiles=False,  # kekulize or not
                                                  rootedAtAtom=-1,  # default
                                                  canonical=False,  # canonicalize or not
                                                  allBondsExplicit=False,  #
                                                  allHsExplicit=False)  #
                    except:
                        smiles = 'None'
                    smiles_list.append(smiles)
            else:
                smiles = 'None'
                smiles_list.append(smiles)
        else:
            try:
                smiles = Chem.MolToSmiles(mol,
                                          isomericSmiles=True,  # keep isomerism
                                          kekuleSmiles=False,  # kekulize or not
                                          rootedAtAtom=-1,  # default
                                          canonical=False,  # canonicalize or not
                                          allBondsExplicit=False,  #
                                          allHsExplicit=False)  #
            except:
                smiles = 'None'
            smiles_list.append(smiles)
        smiles_array = pd.DataFrame(smiles_list).drop_duplicates().values
        # """
        if self.aug_indicator is not None:
            smiles_aug = smiles_array[1:, :]
            np.random.shuffle(smiles_aug)
            smiles_array = np.vstack((smiles_array[0, :], smiles_aug[:self.aug_indicator-1, :]))
        return smiles_array

    """SMILES Augmentation"""
    def smiles_augmentation(self, df):
        column_list = df.columns
        data_aug = np.zeros((1, df.shape[1]))
        for i in range(df.shape[0]):
            smiles = df.iloc[i, 0]
            prop = df.iloc[i, 1:]
            smiles_array = self.generate_smiles(smiles)
            if 'None' not in smiles_array:
                prop = np.tile(prop, (len(smiles_array), 1))
                data_new = np.hstack((smiles_array, prop))
                data_aug = np.vstack((data_aug, data_new))
        df_aug = pd.DataFrame(data_aug[1:, :], columns=column_list)
        return df_aug

    """Used for copolymers with two repeating units"""
    def smiles_augmentation_2(self, df):
        df_columns = df.columns
        column_list = df.columns.tolist()
        column_list_temp = deepcopy(column_list)
        column_list_temp[0] = column_list[1]
        column_list_temp[1] = column_list[0]
        df = df[column_list_temp]
        data_aug = np.zeros((1, df.shape[1]))
        for i in range(df.shape[0]):
            if df.loc[i, "Comonomer percentage"] == 100.0:
                data_new = df.values[i, :].reshape(1, -1)
                data_aug = np.vstack((data_aug, data_new))
            else:
                smiles = df.iloc[i, 0]
                prop = df.iloc[i, 1:]
                smiles_array = self.generate_smiles(smiles)
                if 'None' not in smiles_array:
                    prop = np.tile(prop, (len(smiles_array), 1))
                    data_new = np.hstack((smiles_array, prop))
                    data_aug = np.vstack((data_aug, data_new))
        data_aug_copy = deepcopy(data_aug)
        data_aug_copy[:, 0] = data_aug[:, 1]
        data_aug_copy[:, 1] = data_aug[:, 0]
        df_aug = pd.DataFrame(data_aug_copy[1:, :], columns=df_columns)
        return df_aug

    """Used for combining different repeating unit SMILES"""
    def combine_smiles(self, df):
        for i in range(df.shape[0]):
            if df.loc[i, "Comonomer percentage"] != 100.0:
                df.loc[i, "SMILES descriptor 1"] = df.loc[i, "SMILES descriptor 1"] + '.' + df.loc[
                    i, "SMILES descriptor 2"]
        df = df.drop(columns=['SMILES descriptor 2'])
        return df

    """Combine SMILES with other descriptors to form input sequences"""
    def combine_columns(self, df):
        column_list = df.columns
        for column in column_list[1:-1]:
            df[column_list[0]] = df[column_list[0]] + '$' + df[column].astype("str")
        df = df.drop(columns=df.columns[1:-1].tolist())
        return df