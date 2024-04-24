import os
from collections import Counter
import numpy as np

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

personalities = sorted([
    # Authors
    "William Shakespeare", "Jane Austen", "George Orwell", "J.K. Rowling",
    "Ernest Hemingway", "Mark Twain", "Charles Dickens", "Leo Tolstoy",
    "Agatha Christie", "Virginia Woolf",
    "Haruki Murakami", "Gabriel García Márquez", "Toni Morrison", "Franz Kafka",
    "Fyodor Dostoevsky", "James Baldwin", "Chimamanda Ngozi Adichie", "Salman Rushdie",
    "Octavia E. Butler", "Langston Hughes",

    # Politicians
    "Nelson Mandela", "Winston Churchill", "Margaret Thatcher", "Barack Obama",
    "Mahatma Gandhi", "Angela Merkel", "Abraham Lincoln", "John F. Kennedy",
    "Vladimir Putin", "Xi Jinping",
    "Franklin D. Roosevelt", "Indira Gandhi", "Simón Bolívar", "Benazir Bhutto",
    "Theodore Roosevelt", "Emmanuel Macron", "Jacinda Ardern", "Luiz Inácio Lula da Silva",
    "Aung San Suu Kyi",

    # Musicians
    "Ludwig van Beethoven", "Wolfgang Amadeus Mozart", "Bob Dylan", "The Beatles",
    "Michael Jackson", "Madonna", "Beyoncé", "David Bowie", "Elvis Presley",
    "Freddie Mercury",
    "Prince", "Aretha Franklin", "Johann Sebastian Bach", "Amy Winehouse",
    "Tupac Shakur", "Lady Gaga", "Bob Marley", "Nina Simone", "Jimi Hendrix",
    "Whitney Houston",

    # Historical Figures
    "Albert Einstein", "Martin Luther King Jr.", "Leonardo da Vinci", "Cleopatra",
    "Julius Caesar", "Joan of Arc", "Galileo Galilei", "Isaac Newton",
    "Napoleon Bonaparte", "Alexander the Great",
    "Confucius", "Socrates", "Marie Curie", "Genghis Khan", "Rosa Parks",
    "Queen Elizabeth I", "Charles Darwin", "Harriet Tubman", "Sigmund Freud",
    "Anne Frank",

    # Actors
    "Marilyn Monroe", "Audrey Hepburn", "Marlon Brando", "Meryl Streep",
    "Leonardo DiCaprio", "Denzel Washington", "Tom Hanks", "Natalie Portman",
    "Brad Pitt", "Angelina Jolie",
    "Sidney Poitier", "Cate Blanchett", "Daniel Day-Lewis", "Viola Davis",
    "Heath Ledger", "Charlize Theron", "Joaquin Phoenix", "Lupita Nyong'o",
    "Keanu Reeves", "Saoirse Ronan",

    #Other
    "Simón Bolívar"
])

class TextDataset(Dataset):

    def __init__(self, root_dir, vectorize=1):
        self.labels = []
        self.samples = []
        self.tfidf_vectors = None
        self.bert_vectors = None

        self.vectorize = vectorize
        self.root_dir = root_dir
        self.personalities = personalities

        self._load_dataset()
        if vectorize == 1:
            self._get_bert_embeddings()
        elif vectorize == 2:
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
        label = self.personalities.index(self.labels[idx])
        text = self._read_text(img_path)
        
        if self.vectorize == 1:
            vector = self.bert_vectors[idx]
            return text, label, vector
        elif self.vectorize == 2:
            vector = self.tfidf_vectors.getrow(idx).toarray()[0]
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