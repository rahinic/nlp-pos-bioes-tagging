from torch.utils.data import DataLoader
from dataset import composeDataset
train_dataset = DataLoader(dataset=composeDataset(myDataset="train"),
                                batch_size=64,
                                shuffle=True)
                                
test_dataset = DataLoader(dataset=composeDataset(myDataset="test"),
                                batch_size=64,
                                shuffle=True)

for idx, sample in enumerate(test_dataset):
    print(sample[0][:10])
    print(sample[1][:10])
    if idx>1:
        break


    