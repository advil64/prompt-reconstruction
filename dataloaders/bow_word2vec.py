from dataloaders.dataset_class_word2vec import TextDatasetWord2Vec
from torch.utils.data import DataLoader, random_split

data = TextDatasetWord2Vec('diff-files-new')
train, test = random_split(data, [.7, .3])

train_loader = DataLoader(train, batch_size=100, shuffle=False)
test_loader = DataLoader(test, batch_size=100, shuffle=False)
data_loader = DataLoader(data, batch_size=100, shuffle=False)

for i, (text, labels, vectors) in enumerate(train_loader):
    print(f"Batch {i+1}")
    # print("Text:", text)
    # print("Labels:", labels)
    # print("Vector: ", vectors)
    # print(len(vectors))
    # print("\n")
    # v = vector[0]

    # # Optionally, break after a few batches to avoid flooding the output
    # if i == 10:  # Change this to see more or fewer batches
    #     break