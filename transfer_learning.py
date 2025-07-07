# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load data (numpy arrays)
train_imgs = np.load('C:/Users/dnpdn/OneDrive/Desktop/파/data_files/seg_training.npy')
train_labels = np.load('C:/Users/dnpdn/OneDrive/Desktop/파/data_files/target_labels.npy')
test_imgs = np.load('C:/Users/dnpdn/OneDrive/Desktop/파/data_files/test_images.npy')
test_labels = np.load('C:/Users/dnpdn/OneDrive/Desktop/파/data_files/test_labels.npy')


#print(train_imgs.shape)
#print(train_labels[0])  # one-hot encoded 5 classes

# Convert numpy arrays to PyTorch tensors
train_imgs_tensor = torch.from_numpy(train_imgs).float()
train_labels_tensor = torch.from_numpy(train_labels).float()
test_imgs_tensor = torch.from_numpy(test_imgs).float()
test_labels_tensor = torch.from_numpy(test_labels).float()

# If images are in (N, H, W, C) format, convert to (N, C, H, W) for PyTorch
if train_imgs_tensor.dim() == 4 and train_imgs_tensor.shape[-1] in [1, 3]:
    train_imgs_tensor = train_imgs_tensor.permute(0, 3, 1, 2)
    test_imgs_tensor = test_imgs_tensor.permute(0, 3, 1, 2)


# 클래스 이름 (예: 5개 클래스)
Dict = ['Hat', 'Cube', 'Card', 'Torch', 'screw']

# 1. pretrained VGG16 모델 불러오기 (imagenet 가중치)
model = models.vgg16(pretrained=True)
model.eval()  # 평가 모드로 변경

# 2. 테스트 이미지 (예: test_imgs[idx])를 PIL 이미지로 변환 후 전처리
idx = np.random.randint(len(test_imgs))

# test_imgs[idx]가 numpy 이미지 (H,W,C)라고 가정
img = test_imgs[idx]

# torchvision.transforms로 VGG16이 기대하는 크기와 정규화로 변환
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 평균
                         std=[0.229, 0.224, 0.225])   # ImageNet 표준편차
])

input_tensor = preprocess(img).unsqueeze(0)  # 배치 차원 추가

# 3. 모델에 넣어 예측
with torch.no_grad():
    output = model(input_tensor)

# 4. softmax 확률값 구하기
prob = torch.nn.functional.softmax(output[0], dim=0)

# 5. ImageNet 클래스 라벨 예측 결과 출력 (torchvision 제공)
# (이 부분은 필요시 추가, 별도 클래스 Dict에 맞게 수정 가능)
_, pred_class = torch.max(prob, 0)

print(f"Predicted class index: {pred_class.item()}, Probability: {prob[pred_class].item():.4f}")

# 6. 결과 시각화
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title("Label : {}".format(Dict[np.argmax(train_labels[idx])]))  # train_labels에 맞게 출력
plt.axis('off')
plt.show()

# 7. VGG16 가중치와 편향 가져오기 (예)
vgg16_weights = model.state_dict()

# 예) conv1_1 weight
conv1_1_weight = vgg16_weights['features.0.weight']  # 1번째 conv 층
conv1_1_bias = vgg16_weights['features.0.bias']

# 필요에 따라 변수에 저장 가능
weights = {
    'conv1_1': conv1_1_weight,
    # conv1_2, conv2_1 ... 등 동일하게 features.x.weight 형식으로 접근 가능
}

biases = {
    'conv1_1': conv1_1_bias,
    # conv1_2, conv2_1 ... 등
}

criterion = nn.CrossEntropyLoss()  # softmax 포함한 크로스 엔트로피 손실
optimizer = optim.Adam(vgg16.parameters(), lr=0.001)
def train_batch_maker(batch_size):
    idx = np.random.randint(0, n_train, batch_size)
    return train_imgs_tensor[idx], train_labels_tensor[idx]

def test_batch_maker(batch_size):
    idx = np.random.randint(0, n_test, batch_size)
    return test_imgs_tensor[idx], test_labels_tensor[idx]
n_epoch = 300
n_batch = 20
n_prt = 30

loss_record_train = []

for epoch in range(n_epoch):
    vgg16.train()
    
    inputs, labels = train_batch_maker(n_batch)
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = vgg16(inputs)

    # labels가 one-hot 인코딩이면 정수 레이블로 변환 필요
    if labels.ndim == 2:
        labels = torch.argmax(labels, dim=1)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if epoch % n_prt == 0:
        loss_record_train.append(loss.item())
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

# 학습 손실 그래프
plt.plot(np.arange(len(loss_record_train)) * n_prt, loss_record_train, label='train')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 1. 모델을 평가 모드로 변경
vgg16.eval()

# 2. test_imgs_tensor는 torch.Tensor 타입이고, 이미 device에 맞게 보내졌다고 가정
with torch.no_grad():
    outputs = vgg16(test_imgs_tensor.to(device))  # 전체 테스트 데이터에 대해 예측
    _, predicted = torch.max(outputs, 1)          # 가장 확률 높은 클래스 인덱스 선택

# 3. 정답 라벨이 one-hot encoded 되어 있다면 정수 인덱스로 변환
labels = torch.argmax(test_labels_tensor, dim=1).to(device)

# 4. 정확도 계산
accuracy = (predicted == labels).float().mean().item() * 100
print(f"Accuracy: {accuracy:.2f}%")

# 5. 랜덤 테스트 샘플 하나 선택
idx = np.random.randint(0, test_imgs_tensor.shape[0])
test_x = test_imgs_tensor[idx].unsqueeze(0).to(device)  # 배치 차원 추가
test_y = labels[idx].item()

with torch.no_grad():
    logits = vgg16(test_x)
    probs = torch.softmax(logits, dim=1)
    predict = torch.argmax(probs, dim=1).item()

# 6. 이미지 시각화
plt.figure(figsize=(6, 6))
plt.imshow(test_x.cpu().squeeze().permute(1, 2, 0))  # (C, H, W) -> (H, W, C)
plt.axis('off')
plt.show()

# 7. 예측 결과 및 확률 출력
print(f'Prediction: {Dict[predict]}')
print(f'Probability: {probs.cpu().numpy().ravel()}')
