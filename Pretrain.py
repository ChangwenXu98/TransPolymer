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
torch.cuda.is_available() #checking if CUDA + Colab GPU works

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

    """Set tokenizer"""
    #tokenizer = RobertaTokenizer.from_pretrained("roberta-base",max_len=512)
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=pretrain_config['blocksize'])

    """Construct MLM model"""
    model = RobertaForMaskedLM(config=config).to(device)

    """Load Data"""
    train_data, valid_data = split(pretrain_config['file_path'])
    data_train = LoadPretrainData(tokenizer=tokenizer, dataset=train_data, blocksize=pretrain_config['blocksize'])
    data_valid = LoadPretrainData(tokenizer=tokenizer, dataset=valid_data, blocksize=pretrain_config['blocksize'])

    """Set DataCollator"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=pretrain_config['mlm_probability']
    )

    """Training arguments"""
    training_args = TrainingArguments(
        output_dir=pretrain_config['save_path'],
        overwrite_output_dir=pretrain_config['overwrite_output_dir'],
        num_train_epochs=pretrain_config['epochs'],
        per_device_train_batch_size=pretrain_config['batch_size'],
        per_device_eval_batch_size=pretrain_config['batch_size'],
        save_strategy=pretrain_config['save_strategy'],
        save_total_limit=pretrain_config['save_total_limit'],
        fp16=pretrain_config['fp16'],
        logging_strategy=pretrain_config['logging_strategy'],
        evaluation_strategy=pretrain_config['evaluation_strategy'],
        learning_rate=pretrain_config['lr_rate'],
        lr_scheduler_type=pretrain_config['scheduler_type'],
        weight_decay=pretrain_config['weight_decay'],
        warmup_ratio=pretrain_config['warmup_ratio'],
        report_to=pretrain_config['report_to'],
        dataloader_num_workers=pretrain_config['dataloader_num_workers'],
        sharded_ddp=pretrain_config['sharded_ddp'],
    )

    """Set Trainer"""
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data_train,
        eval_dataset=data_valid
    )

    """
    writer = SummaryWriter(log_dir=training_args.logging_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', training_args.logging_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    """
    

    """Train and save model"""
    #torch.cuda.empty_cache()
    #trainer.train()
    trainer.train(resume_from_checkpoint=pretrain_config['load_checkpoint'])
    trainer.save_model(pretrain_config["save_path"])

if __name__ == "__main__":

    pretrain_config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    """Run the main function"""
    main(pretrain_config)