from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets

tranform = ToTensor()
train_data = datasets.FashionMNIST(root='Fashion_mnist',train=True,transform=tranform,download=False)
test_data = datasets.FashionMNIST(root='Fashion_mnist',train=False,transform=tranform,download=False)

batch_sz= 10
# train_loader = DataLoader(train_data,batch_size=batch_sz,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_sz,shuffle= False)