
from torch.utils.data import DataLoader
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np


def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(0)
np.random.seed(0)


BATCH_SIZE = 32
LR = 5e-5
NUM_EPOCHES = 300

mean, std = (0.5,), (0.5,)


transform_train = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(degrees=(-90, 90)),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.Normalize(mean, std)
                              ])

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                              ])


trainset = datasets.CIFAR10('../data/CIFAR10/', download=True, train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = datasets.CIFAR10('../data/CIFAR10/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

teacher_model = torch.load("../teacher_model/best_Regnety160-8.pth")

teacher_model.preprocess_flag = False
print(teacher_model.preprocess_flag)

from transformer_package.models import DeiT

image_size = 224
channel_size = 3
patch_size = 4
embed_size = 512
num_heads = 8
classes = 10
num_layers = 4
hidden_size = 512
dropout = 0.2


model = DeiT(image_size=image_size,
             channel_size=channel_size,
             patch_size=patch_size,
             embed_size=embed_size,
             num_heads=num_heads,
             classes=classes,
             num_layers=num_layers,
             hidden_size=hidden_size,
             teacher_model=teacher_model,
             dropout=dropout
            ).to(device)
model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
model = model.to(device)
from transformer_package.loss_functions.loss import Soft_Distillation_Loss

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

loss_hist = {}
loss_hist["train accuracy"] = []
loss_hist["train loss"] = []

h_acc = 0

temperature = 1 #Soft Distillation Temperature


for epoch in range(1, NUM_EPOCHES + 1):
    model.train()

    epoch_train_loss = 0

    y_true_train = []
    y_pred_train = []



    for batch_idx, (img, labels) in enumerate(trainloader):
        img = img.to(device)
        labels = labels.to(device)

        preds, teacher_preds = model(img)

        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
        y_true_train.extend(labels.detach().tolist())

        epoch_train_loss += loss.item()

    loss_hist["train loss"].append(epoch_train_loss)

    total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x == y])
    total = len(y_pred_train)
    accuracy = total_correct * 100 / total

    loss_hist["train accuracy"].append(accuracy)



    print("-------------------------------------------------")
    print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
    print("       Train Accuracy%: ", accuracy, "==", total_correct, "/", total)


    with torch.no_grad():
        model.eval()

        y_true_test = []
        y_pred_test = []

        for batch_idx, (img, labels) in enumerate(testloader):
            img = img.to(device)
            labels = labels.to(device)

            preds = model(img)

            y_pred_test.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_test.extend(labels.detach().tolist())

        total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x == y])
        total = len(y_pred_test)
        accuracy = total_correct * 100 / total
        if h_acc <= accuracy:
            save_checkpoint(model, f'Pretrained_best_DeiT(None).pth')
            h_acc = accuracy

        print("Test Accuracy%: ", accuracy, "==", total_correct, "/", total)
        print("-------------------------------------------------")