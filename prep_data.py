import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import pdb

def only_smiles(smiles_file, output_file):
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_smiles[['solv_comb_sm', 'salt_sm']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_smiles['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def smiles_and_temp(num_file, smiles_file, output_file):
    df_smiles = pd.read_csv(smiles_file)
    df_num = pd.read_csv(num_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input = df_input.applymap(str)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_comb_sm', 'salt_sm', 'temperature']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_smiles['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def exp_2():
    df_num = pd.read_csv("./data/all_multi_comp_add.csv")
    df_smiles = pd.read_csv("./data/all_multi_comp.csv")
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input = df_input.applymap(str)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4', 'conc_salt', 'temperature', 'solv_1_sm', 'solv_2_sm', 'solv_3_sm','solv_4_sm', 'salt_sm']].apply("$".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']

    train, test = train_test_split(df_input_final, test_size=0.2)

    print(train.shape[0], test.shape[0])

    df_input_final.to_csv("./data/our_combined.csv", index = False)
    train.to_csv("./data/our_train_combined.csv", index = False)
    test.to_csv("./data/our_test_combined.csv", index = False)

def exp_3(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input = df_input.applymap(str)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4', 'conc_salt', 'temperature', 'solv_comb_sm', 'salt_sm']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def exp_4(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input = df_input.applymap(str)   
    df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1', 'mol_wt_solv_1']].apply("$".join, axis=1)
    df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2', 'mol_wt_solv_2']].apply("$".join, axis=1)
    df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3', 'mol_wt_solv_3']].apply("$".join, axis=1)
    df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4', 'mol_wt_solv_4']].apply("$".join, axis=1)
    df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt', 'temperature']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']
    df_input_final.to_csv(output_file, index = False)


def main():
    exp_3( "./data/freqII_train_multi_comp_add.csv", "./data/freqII_train_multi_comp_comb.csv", "./data/freqII_train_all_pipe.csv")

if __name__ == "__main__":
    main()