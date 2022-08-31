# TransPolymer

This is an unofficial introduction to the datasets and codes used in the paper "TransPolymer: a Transformer-based Language Model for Polymer Property Predictions". In this work, we introduce TransPolymer, a Transformer-based language model, for representation learning of polymer sequences by pretraining on a large unlabeled dataset via self-supervised masked language modeling and making accurate and efficient predictions of polymer properties in downstream tasks by finetuning. 

# Datasets

The pretraining dataset is adopted from the paper ["PI1M: A Benchmark Database for Polymer Informatics"](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00726). Data augmentation is applied by augmenting each sequence to five.

Eight datasets, concerning different polymer properties including polymer electrolyte conductivity, band gap, crystallization tendency, dielectric constant, ionization energy, refractive index, and p-type polymer OPV power conversion efficiency, are used for downstream tasks. Data processing and augmentation are implemented before usage in the finetuning stage. The original datasets and their sources are listed below:

PE-I: ["AI-Assisted Exploration of Superionic Glass-Type Li(+) Conductors with Aromatic Structures"](https://pubs.acs.org/doi/10.1021/jacs.9b11442)

PE-II: ["Database Creation, Visualization, and Statistical Learning for Polymer Li+-Electrolyte Design"](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.0c04767)

Egc, Xc, EPS, Ei, Nc: ["Polymer informatics with multi-task learning"](https://www.sciencedirect.com/science/article/pii/S2666389921000581)

OPV: ["Computer-Aided Screening of Conjugated Polymers for Organic Solar Cell: Classification by Random Forest"](https://pubs.acs.org/doi/10.1021/acs.jpclett.8b00635)

The processed datasets will be made public as soon as the paper is published.

# Codes

The codes used for pretraining, finetuning, and visualization are included in the archive. `PolymerSmilesTokenization.py` is adapted from RobertaTokenizer from [huggingface](https://github.com/huggingface/transformers/tree/v4.21.2) with a specially designed regular expression for tokenization with chemical awareness. `Pretrain.py` is used for pretraining. `Downstream_train_test_split.py` is used for finetuning on PE-I dataset which adopts a train-test split, and `Downstream_with_aug.py` is used for the other datasets except PE-II which use cross-validation. `Downstream_with_aug_special.py` is specially for PE-II which also uses cross-validation but the data augmentation function is more complex due to the copolymers contained in the dataset. Correspondingly, `Predict_vs_true.py`, `Predict_vs_true_aug.py`, and `Predict_vs_true_special.py` are used to draw plots of ground truth vs. predicted value for the downstream tasks, respectively. `Attention_vis_pretrain.py` is used for attention visualization of pretrained TransPolymer, and `Attention_vis_pred.py` is used for visualizing the determinant tokens in polymer sequences on prediction results. `Obtain_Embedding.py` is used for extracting TransPolymer embeddings from sequences and apply max pooling on them as input of t-SNE, and t-SNE visualization is achieved by `tSNE.py`. The plots of length distributions of polymer sequences in downstream datasets before data augmentation are drawn by `length_dist.py`.
