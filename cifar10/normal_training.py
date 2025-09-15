import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
import os
from resnet_architecture import ResNet18
from training_routines import copy_state_dict, accuracy_evaluation, train_no
from datetime import datetime


################################ Loading data ####################################

data_dir = 'dataset'
train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=False)
test_dataset  = torchvision.datasets.CIFAR10(data_dir, train=False, download=False)

train_transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                 (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
                                ])

test_transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])


test_dataset.transform = test_transform
train_dataset.transform = train_transform

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,shuffle=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True)


############################ Initializing ResNet18 Model and training params #############################

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = ResNet18(num_classes=10,inter_dim=128).to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)


########################## Start Training ##############################################
print("Staring training....")
now = datetime.now()
epochs = 2
correct = 0
epoch_max = 0

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    _train_batch_loss , _train_batch_accuracy = train_no(train_loader, model, loss_fn, optimizer,device)
    correct_temp = accuracy_evaluation(test_loader, model,device)
    print(correct_temp)
    
    if correct_temp >= correct:
        state = copy_state_dict(model)
        correct = correct_temp
        epoch_max = t
        
    
    scheduler.step()
    print('Time taken:', datetime.now()-now)


if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')

torch.save(state, f'./saved_models/resnet18_sgd_train_no_200_epochs')


print("Done!")
print(datetime.now())
print('Time taken:', datetime.now()-now)














