import os
import chardet
import numpy as np
from torch.utils.data import Dataset
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
import nltk
import torch
import nltk
import unicodedata
from nltk.corpus import stopwords
nltk.download('stopwords')


nltk.download('punkt')  # Make sure to have the punkt tokenizer downloaded

class TextDatasetDoc2Vec(Dataset):
    def __init__(self, root_dir, train_doc2vec=True, vector_size=100, window=5, min_count=1, workers=4):
        self.samples = []
        self.labels = []
        self.root_dir = root_dir
        self.documents = []  # Collect documents for Doc2Vec training
        self.vector_size = vector_size

        self._load_dataset()

        if train_doc2vec:
            self._train_doc2vec(vector_size, window, min_count, workers)

    def _load_dataset(self):
        # Assuming a simple structure where each file is a separate document and its name before the extension is its label
        for folder_dir in os.listdir(self.root_dir):
            for filename in os.listdir(os.path.join(self.root_dir, folder_dir)):
                file_path = os.path.join(self.root_dir, folder_dir, filename)
                if os.path.isfile(file_path):
                    text = self._read_text(file_path)
                    self.samples.append(file_path)
                    self.labels.append(folder_dir)
                    # Each document needs to be a TaggedDocument for training Doc2Vec
                    cleaned_text = self._clean_text(text)
                    self.documents.append(TaggedDocument(words=simple_preprocess(cleaned_text), tags=[folder_dir]))

    # def _load_dataset(self):
    #     # Assuming a simple structure where each file is a separate document and its name before the extension is its label
    #     #print("self.root_dir: ", self.root_dir)
    #     #print("os.listdir: ", os.listdir(self.root_dir))
    #     for folder_dir in os.listdir(self.root_dir):
    #         #print("folder_dir: ", folder_dir)
    #         #print("os.listdir(folder_dir): ", os.listdir(f"{self.root_dir}/{folder_dir}"))
    #         for filename in os.listdir(f"{self.root_dir}/{folder_dir}"):
    #             file_path = os.path.join(f"{self.root_dir}/{folder_dir}", filename)
    #             if os.path.isfile(file_path):
    #                 text = self._read_text(file_path)
    #                 self.samples.append(file_path)
    #                 self.labels.append(folder_dir)
    #                 self.texts.append(text)

    def _read_text(self, file_path) -> str:
        with open(file_path, 'rb') as file:  # Open the file in binary mode
            raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding']

        with open(file_path, 'r', encoding="utf8") as file:
            return file.read()

    def _train_doc2vec(self, vector_size, window, min_count, workers):
        self.doc_vectors = Doc2Vec(documents=self.documents, vector_size=vector_size, window=window,
                                   min_count=min_count, workers=workers)

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
        # Convert text to lowercase
        text = text.lower()
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        # Remove punctuation
        tokens = [word for word in tokens if word.isalpha()]
        # Remove pronunciation marks and combine into a single string
        cleaned_text = ' '.join(tokens)
        cleaned_text = unicodedata.normalize('NFKD', cleaned_text).encode('ASCII', 'ignore').decode('utf-8')
        return cleaned_text



