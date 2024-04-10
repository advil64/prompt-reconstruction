import os

import chardet
import numpy as np
from torch.utils.data import Dataset
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')  # Make sure to have the punkt tokenizer downloaded


class TextDatasetWord2Vec(Dataset):
    def __init__(self, root_dir, train_word2vec=True, vector_size=1000, window=5, min_count=1, workers=4):
        self.samples = []
        self.labels = []
        self.root_dir = root_dir
        self.texts = []  # Collect texts for Word2Vec training

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

# import os
# from torch.utils.data import Dataset
# from gensim.models import KeyedVectors
# import numpy as np
#
# class TextDatasetWord2Vec(Dataset):
#
#     def __init__(self, root_dir, vectorize=False, word2vec_path='GoogleNews-vectors-negative300.bin'):
#         """
#         :param root_dir: Directory with all the text files.
#         :param vectorize: Boolean to decide whether to vectorize text using Word2Vec.
#         :param word2vec_path: Path to a pre-trained Word2Vec model.
#         """
#         self.labels = []
#         self.samples = []
#         self.vectorize = vectorize
#         self.word_vectors = None
#
#         if vectorize and word2vec_path:
#             self.word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
#             print("CHECK1")
#
#         print(self.word_vectors)
#
#         self.root_dir = root_dir
#         self._load_dataset()
#
#     def _load_dataset(self):
#         for label in os.listdir(self.root_dir):
#             class_dir = os.path.join(self.root_dir, label)
#             if os.path.isdir(class_dir):
#                 for filename in os.listdir(class_dir):
#                     file_path = os.path.join(class_dir, filename)
#                     if os.path.isfile(file_path):
#                         self.samples.append(file_path)
#                         self.labels.append(label)
#
#     def _read_text(self, file_path) -> str:
#         with open(file_path, 'r') as file:
#             return file.read()
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         file_path = self.samples[idx]
#         label = self.labels[idx]
#         text = self._read_text(file_path)
#
#         if self.vectorize and self.word_vectors:
#             # Average Word2Vec over all words in the document
#             words = text.split()
#             vectors = [self.word_vectors[word] for word in words if word in self.word_vectors]
#             if vectors:
#                 vector = np.mean(vectors, axis=0)
#             else:
#                 # Handle case where none of the words are in the model's vocabulary
#                 vector = np.zeros(self.word_vectors.vector_size)
#             return text, label, vector
#
#         return text, label
