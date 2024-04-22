import os
import random
import unicodedata

import chardet
import nltk
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from torch.utils.data import Dataset

nltk.download('stopwords')
nltk.download('punkt')

class TextDatasetDoc2Vec(Dataset):
    def __init__(self, root_dir, use_pretrained=True, vector_size=1000, window=5, min_count=1, workers=4, epochs=50):
        self.samples = []
        self.labels = []
        self.root_dir = root_dir
        self.documents = []  # Collect documents for Doc2Vec training
        self.vector_size = vector_size
        self.epochs = epochs

        # Load dataset and prepare training data
        self._load_dataset()

        # Train Doc2Vec model on the complete dataset
        if not use_pretrained:
            self._train_doc2vec(vector_size, window, min_count, workers, epochs)

    def _load_dataset(self):
        for folder_dir in os.listdir(self.root_dir):
            for filename in os.listdir(os.path.join(self.root_dir, folder_dir)):
                file_path = os.path.join(self.root_dir, folder_dir, filename)
                if os.path.isfile(file_path):
                    text = self._read_text(file_path)
                    self.samples.append(file_path)
                    self.labels.append(folder_dir)
                    cleaned_text = self._clean_text(text)
                    self.documents.append(TaggedDocument(words=simple_preprocess(cleaned_text), tags=[folder_dir]))

    def _read_text(self, file_path) -> str:
        with open(file_path, 'rb') as file:  # Open the file in binary mode
            raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding']

        with open(file_path, 'r', encoding="utf8") as file:
            return file.read()

    def _train_doc2vec(self, vector_size, window, min_count, workers, epochs):
        self.doc_vectors = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)
        self.doc_vectors.build_vocab(self.documents)
        self.doc_vectors.train(self.documents, total_examples=self.doc_vectors.corpus_count, epochs=self.doc_vectors.epochs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]
        text = self._read_text(file_path)
        cleaned_text = self._clean_text(text)
        if hasattr(self, 'doc_vectors'):
            vector = self.doc_vectors.infer_vector(simple_preprocess(cleaned_text))
            return text, label, vector
        return text, label

    def _clean_text(self, text):
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = [word for word in tokens if word.isalpha()]
        cleaned_text = ' '.join(tokens)
        cleaned_text = unicodedata.normalize('NFKD', cleaned_text).encode('ASCII', 'ignore').decode('utf-8')
        return cleaned_text
