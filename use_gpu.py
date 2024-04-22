from dataloaders.dataset_class_bert import TextDataset
from torch.utils.data import DataLoader

data = TextDataset('/common/home/ac1771/Desktop/prompt-reconstruction/data/diff-files-final', vectorize=True)
train_loader = DataLoader(data, batch_size=32, shuffle=True)