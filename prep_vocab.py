import pandas as pd
import pdb
import csv

df_PE_I_vocab = pd.read_csv('./data/vocab/vocab_sup_PE_I.csv')
df_freq_II_smiles = pd.read_csv('./data/freqII_train_multi_comp.csv')
df_freq_II_nums = pd.read_csv('./data/freqII_train_multi_comp_add.csv')

new_vocab = []

unique_PE_I = set(df_PE_I_vocab['$'])
unique_PE_I.add('$')
unique_conductivity = set(df_freq_II_smiles['conductivity_log'])

# pdb.set_trace()
new_vocab = unique_PE_I | unique_conductivity

print(len(new_vocab))

num_columns = df_freq_II_nums.shape[1]

unique_nums = []

for i in range(num_columns):
    column = df_freq_II_nums.iloc[:, i]
    new_vocab = new_vocab | set(column)

print(len(new_vocab))

df = pd.DataFrame(new_vocab)

df.to_csv('./data/vocab/vocab_freqII_train.csv', index=False)
