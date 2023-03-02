from __future__ import print_function

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
import torchvision.transforms as transforms

use_cuda=True

epsilons = [ 0.1]


# MNIST 테스트 데이터셋과 데이터로더 선언
trans = transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            ])
# train_set = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=trans)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=trans)

# trainloader = torch.utils.data.DataLoader(
#     train_set, batch_size=100, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, num_workers=0)

# 어떤 디바이스를 사용할지 정의
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

model_path = "../ResNet-50/best_resnet50.pth"
model = torch.load(model_path)

model.eval()
if torch.cuda.is_available():
    model.cuda()


def pgd_attack(model, images, labels, eps=0.1, alpha=2 / 255, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

correct = 0
label_class = "hello"
adv_examples = []
loop = tqdm(enumerate(test_loader), total=len(test_loader), mininterval=0.01, leave=True)
img_len = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i, data in loop:
    inputs, labels = data
    # inputs, labels = inputs.to(device), labels.to(device)
    inputs = torch.clamp(inputs, 0, 1)

    if str(labels.item()) == "0":
        if img_len[0] < 100:
            label_class = "airplane"
            img_len[0] += 1
        else:
            continue
    elif str(labels.item()) == "1":
        if img_len[1] < 100:
            label_class = "automobile"
            img_len[1] += 1
        else:
            continue
    elif str(labels.item()) == "2":
        if img_len[2] < 100:
            label_class = "bird"
            img_len[2] += 1
        else:
            continue
    elif str(labels.item()) == "3":
        if img_len[3] < 100:
            label_class = "cat"
            img_len[3] += 1
        else:
            continue
    elif str(labels.item()) == "4":
        if img_len[4] < 100:
            label_class = "deer"
            img_len[4] += 1
        else:
            continue
    elif str(labels.item()) == "5":
        if img_len[5] < 100:
            label_class = "dog"
            img_len[5] += 1
        else:
            continue
    elif str(labels.item()) == "6":
        if img_len[6] < 100:
            label_class = "frog"
            img_len[6] += 1
        else:
            continue
    elif str(labels.item()) == "7":
        if img_len[7] < 100:
            label_class = "horse"
            img_len[7] += 1
        else:
            continue
    elif str(labels.item()) == "8":
        if img_len[8] < 100:
            label_class = "ship"
            img_len[8] += 1
        else:
            continue
    elif str(labels.item()) == "9":
        if img_len[9] < 100:
            label_class = "truck"
            img_len[9] += 1
        else:
            continue

    adv_images = pgd_attack(model, inputs, labels)
    output = model(adv_images)

    adv_images = adv_images.cpu().detach().numpy()
    adv_images = adv_images.squeeze()
    plt.imshow(np.transpose(adv_images, (1, 2, 0)))
    plt.axis('off')
    plt.savefig('C:/ResNet_model_PGD/test/0.1/' + label_class + '/' + str(i) + '.png',
                bbox_inches='tight', pad_inches=0)
