from __future__ import print_function

from torch.utils.data import DataLoader
import torchvision
import torch

import torch.nn.functional as F
from torchvision import  transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# train 생성
epsilons = [ 0.1]

use_cuda=True


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

# 모델 초기화하기
model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
path = "../DeiT/Pretrained_best_DeiT.pth"
model.load_state_dict(torch.load(path))
model.eval()
if torch.cuda.is_available():
    model.cuda()

# FGSM 공격 코드
def fgsm_attack(image, epsilon, data_grad):

    # data_grad 의 요소별 부호 값을 얻어옵니다
    sign_data_grad = data_grad.sign()
    # 입력 이미지의 각 픽셀에 sign_data_grad 를 적용해 작은 변화가 적용된 이미지를 생성합니다
    perturbed_image = image + epsilon*sign_data_grad
    # 값 범위를 [0,1]로 유지하기 위해 자르기(clipping)를 추가합니다
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 작은 변화가 적용된 이미지를 리턴합니다
    return perturbed_image




def test( model, device, test_loader, epsilon ):
    label_class = "hello"
    # 정확도 카운터
    correct = 0
    adv_examples = []
    img_len = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    loop = tqdm(enumerate(test_loader), total=len(test_loader))
    # 테스트 셋의 모든 예제에 대해 루프를 돕니다
    for i, data in loop:
        # 디바이스(CPU or GPU) 에 데이터와 라벨 값을 보냅니다
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 텐서의 속성 중 requires_grad 를 설정합니다. 공격에서 중요한 부분입니다
        inputs.requires_grad = True

        # 데이터를 모델에 통과시킵니다
        output = model(inputs)
        init_pred = output.max(1, keepdim=True)[1] # 로그 확률의 최대값을 가지는 인덱스를 얻습니다
        init_pred = init_pred.squeeze()


        # 손실을 계산합니다
        loss = F.nll_loss(output, labels)
        # 모델의 변화도들을 전부 0으로 설정합니다
        model.zero_grad()
        # 후방 전달을 통해 모델의 변화도를 계산합니다
        loss.backward()
        # 변화도 값을 모읍니다
        data_grad = inputs.grad
        # FGSM 공격을 호출합니다

        perturbed_data = fgsm_attack(inputs, epsilon, data_grad)
        # 작은 변화가 적용된 이미지에 대해 재분류합니다
        output = model(perturbed_data)
        # if epsilon == 0:
        #     output = model(inputs)

        # 올바른지 확인합니다
        final_pred = output.max(1, keepdim=True)[1] # 로그 확률의 최대값을 가지는 인덱스를 얻습니다

        if final_pred.squeeze() == labels.squeeze():
            correct += 1

        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()



        if str(labels) == "tensor([0], device='cuda:0')":
            if img_len[0] < 100:
                label_class = "airplane"
                img_len[0] += 1
            else:
                continue
        elif str(labels) == "tensor([1], device='cuda:0')":
            if img_len[1] < 100:
                label_class = "automobile"
                img_len[1] += 1
            else:
                continue
        elif str(labels) == "tensor([2], device='cuda:0')":
            if img_len[2] < 100:
                label_class = "bird"
                img_len[2] += 1
            else:
                continue
        elif str(labels) == "tensor([3], device='cuda:0')":
            if img_len[3] < 100:
                label_class = "cat"
                img_len[3] += 1
            else:
                continue
        elif str(labels) == "tensor([4], device='cuda:0')":
            if img_len[4] < 100:
                label_class = "deer"
                img_len[4] += 1
            else:
                continue
        elif str(labels) == "tensor([5], device='cuda:0')":
            if img_len[5] < 100:
                label_class = "dog"
                img_len[5] += 1
            else:
                continue
        elif str(labels) == "tensor([6], device='cuda:0')":
            if img_len[6] < 100:
                label_class = "frog"
                img_len[6] += 1
            else:
                continue
        elif str(labels) == "tensor([7], device='cuda:0')":
            if img_len[7] < 100:
                label_class = "horse"
                img_len[7] += 1
            else:
                continue
        elif str(labels) == "tensor([8], device='cuda:0')":
            if img_len[8] < 100:
                label_class = "ship"
                img_len[8] += 1
            else:
                continue
        elif str(labels) == "tensor([9], device='cuda:0')":
            if img_len[9] < 100:
                label_class = "truck"
                img_len[9] += 1
            else:
                continue


        # adv_ex = adv_ex / 2 + 0.5
        plt.imshow(np.transpose(adv_ex, (1, 2, 0)))
        plt.axis('off')
        plt.savefig('C:/original/' + str(epsilon) + '/' + label_class + '/' + str(i) + '.png', bbox_inches='tight', pad_inches=0)

    # 해2+당 엡실론에서의 최종 정확도를 계산합니다
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # 정확도와 적대적 예제를 리턴합니다
    return final_acc

accuracies = []
examples = []

# 각 엡실론에 대해 테스트 함수를 실행합니다
for eps in epsilons:
    acc = test(model, device, test_loader, eps)
    accuracies.append(acc)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()



# # 각 엡실론에서 적대적 샘플의 몇 가지 예를 도식화합니다
# for i in tqdm(range(len(epsilons)), desc="epsilon generation", leave=True):
#     print("epsilon"+str(epsilons[i])+" generation start")
#     for j in tqdm(range(len(examples[i])), desc="class generation", leave=True):
#         orig, adv, ex, label = examples[i][j]
#         if str(label) == "tensor([0], device='cuda:0')":
#             label_class = "dyed-lifted-polyps"
#         elif str(label) == "tensor([1], device='cuda:0')":
#             label_class = "dyed-resection-margins"
#         elif str(label) == "tensor([2], device='cuda:0')":
#             label_class = "esophagitis"
#         elif str(label) == "tensor([3], device='cuda:0')":
#             label_class = "normal-cecum"
#         elif str(label) == "tensor([4], device='cuda:0')":
#             label_class = "normal-pylorus"
#         elif str(label) == "tensor([5], device='cuda:0')":
#             label_class = "normal-z-line"
#         elif str(label) == "tensor([6], device='cuda:0')":
#             label_class = "polyps"
#         elif str(label) == "tensor([7], device='cuda:0')":
#             label_class = "ulcerative-colitis"
#         ex = ex / 2 + 0.5
#         print(ex)
#         plt.imshow(np.transpose(ex, (1, 2, 0)))
#         plt.axis('off')
#         plt.savefig('C:/test/'+str(i)+str(j)+'.png', bbox_inches='tight', pad_inches=0)


