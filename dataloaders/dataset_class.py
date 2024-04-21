import os
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class TextDataset(Dataset):

    def __init__(self, root_dir, vectorize=False):
        self.labels = []
        self.samples = []
        self.tfidf_vectors = None

        self.vectorize = vectorize
        self.root_dir = root_dir

        self._load_dataset()
        if vectorize:
            self._build_vocab_and_idf()

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
            vector = self.tfidf_vectors.getrow(idx).toarray()[0]
            return text, label, vector
        
        return text, label
    
    def _build_vocab_and_idf(self):
        vectorizer = TfidfVectorizer(input='filename', stop_words='english')
        self.tfidf_vectors = vectorizer.fit_transform(self.samples)