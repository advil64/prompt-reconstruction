import os

import chardet
import nltk
import numpy as np
import torch
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset

nltk.download('punkt')  # Make sure to have the punkt tokenizer downloaded


class TextDatasetWord2Vec(Dataset):
    def __init__(self, root_dir, train_word2vec=True, vector_size=100, window=5, min_count=1, workers=4):
        self.samples = []
        self.labels = []
        self.root_dir = root_dir
        self.texts = []  # Collect texts for Word2Vec training
        self.vector_size = vector_size

        self._load_dataset()

        if train_word2vec:
            self._train_word2vec(vector_size, window, min_count, workers)

    def _load_dataset(self):
        # Assuming a simple structure where each file is a separate document and its name before the extension is its label
        #print("self.root_dir: ", self.root_dir)
        #print("os.listdir: ", os.listdir(self.root_dir))
        for folder_dir in os.listdir(self.root_dir):
            #print("folder_dir: ", folder_dir)
            #print("os.listdir(folder_dir): ", os.listdir(f"{self.root_dir}/{folder_dir}"))
            for filename in os.listdir(f"{self.root_dir}/{folder_dir}"):
                file_path = os.path.join(f"{self.root_dir}/{folder_dir}", filename)
                if os.path.isfile(file_path):
                    text = self._read_text(file_path)
                    self.samples.append(file_path)
                    self.labels.append(folder_dir)
                    self.texts.append(text)

    import chardet

    def _read_text(self, file_path) -> str:
        with open(file_path, 'rb') as file:  # Open the file in binary mode
            raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding']

        # Now open the file with the detected encoding
        with open(file_path, 'r', encoding="utf8") as file:
            return file.read()

    def _train_word2vec(self, vector_size, window, min_count, workers):
        tokenized_texts = [simple_preprocess(sent) for text in self.texts for sent in sent_tokenize(text)]
        self.word_vectors = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window,
                                     min_count=min_count, workers=workers).wv

    def __len__(self):
        return len(self.samples)

    #Original get item with vector as average of word vectors
    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]
        text = self._read_text(file_path)

        if hasattr(self, 'word_vectors') and self.word_vectors:
            # Vectorize text using the trained Word2Vec model
            words = simple_preprocess(text)
            vectors = [self.word_vectors[word] for word in words if word in self.word_vectors]
            if vectors:
                vector = np.mean(vectors, axis=0)
            else:
                vector = np.zeros(self.vector_size)
            return text, label, vector

        return text, label

    # vector for each word (NOT WORKING)
    # def __getitem__(self, idx):
    #     file_path = self.samples[idx]
    #     label = self.labels[idx]
    #     text = self._read_text(file_path)

    #     if hasattr(self, 'word_vectors') and self.word_vectors:
    #         # Vectorize text using the trained Word2Vec model
    #         words = simple_preprocess(text)
    #         vectors = [self.word_vectors[word] for word in words if word in self.word_vectors]

    #         # Pad vectors to the same length
    #         max_length = max(len(vec) for vec in vectors)
    #         padded_vectors = []
    #         for vec in vectors:
    #             pad_length = max_length - len(vec)
    #             if pad_length > 0:
    #                 zero_pad = torch.zeros((pad_length, self.vector_size), dtype=torch.float32)
    #                 padded_vec = torch.cat([torch.tensor(vec, dtype=torch.float32), zero_pad], dim=0)
    #             else:
    #                 padded_vec = torch.tensor(vec, dtype=torch.float32)
    #             padded_vectors.append(padded_vec)

    #         # Stack padded vectors into a tensor
    #         vector = torch.stack(padded_vectors)

    #         return text, label, vector

    #     return text, label
