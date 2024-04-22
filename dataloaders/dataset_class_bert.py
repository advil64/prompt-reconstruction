import os
from collections import Counter
import numpy as np

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class TextDataset(Dataset):

    def __init__(self, root_dir, vectorize=False):
        self.labels = []
        self.samples = []
        self.tfidf_vectors = None
        self.bert_vectors = None

        self.vectorize = vectorize
        self.root_dir = root_dir

        self._load_dataset()
        if vectorize:
            self._get_bert_embeddings()

    def _load_dataset(self):
        for label in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, filename)
                    if os.path.isfile(file_path):
                        self.samples.append(file_path)
                        self.labels.append(label)
    
    def _read_text(self, file_path) -> str:
        with open(file_path, 'r') as file:
            return file.read()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        text = self._read_text(img_path)
        
        if self.vectorize:
            vector = self.bert_vectors[idx]
            return text, label, vector
        
        return text, label
    
    def _get_bert_embeddings(self):

        if os.path.exists('bert_embeddings.npy'):
            bert_embeddings = np.load('bert_embeddings.npy')
            self.bert_vectors = bert_embeddings
            return
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model = model.to(device)

        batch_size = 16  # Define a reasonable batch size for your GPU
        num_samples = len(self.samples)
        all_embeddings = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_samples = self.samples[start_idx:end_idx]
            encoded_inputs = tokenizer(batch_samples, padding=True, truncation=True, return_tensors='pt')
            encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}

            with torch.no_grad():
                outputs = model(**encoded_inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())

        self.bert_vectors = np.concatenate(all_embeddings, axis=0)
        np.save('bert_embeddings.npy', self.bert_vectors)
    
    def _build_vocab_and_idf(self):
        vectorizer = TfidfVectorizer(input='filename', stop_words='english')
        self.tfidf_vectors = vectorizer.fit_transform(self.samples)