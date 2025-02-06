# import time
# import torch
# from torch.utils.data import DataLoader
# from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import AutoConfig, BertForMaskedLM
# from dataset import ProteinGoDataset, ProteinSeqDataset, GoGoDataset
# from sampling import negative_sampling_strategy
# from dataloader import DataCollatorForGoGo, DataCollatorForProteinGo, DataCollatorForLanguageModeling


if __name__ == '__main__':

    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification.from_pretrained('/OntoGene/data/output_data/checkpoint-1171')
    