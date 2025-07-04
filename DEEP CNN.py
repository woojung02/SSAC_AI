import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = torch.Tensor(1, 1, 28, 28)##(배치크기,채널,높이,너비)
conv1 = nn.Conv2d(1, 32, 3, padding=1)##합성곱층와 풀링선언(1채널 입력 받아 32채널 나오고 커널 사이즈는3*3,가로 세로 한줄씩 패딩)
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)##두 번째 합성곱층(32채널 입력 받아서,64개 채널 나오고 ,커널 사이즈3*3,한줄씩 패딩)
conv3=nn.Conv2d(64,128,3,1)#3번째 합성곱층(input=64,output=128,커널 사이즈3,한줄씩 패딩)
pool = nn.MaxPool2d(2)##MAXPOOING 커널 사이즈 2,스트라이드 2

out = conv1(inputs)#연결하기1(1,32,28,28)
out = pool(out)#연결하기2(1,32,14,14)

out = conv2(out)#연결하기3(1,64,14,14)
out = pool(out)#연결하기4(1,64,7,7)

out=conv3(out)#연결하기5(1,128,5,5)
out=pool(out)#연결하기6(1,128,2,2)

out = out.view(out.size(0), -1) #첫 번째 차원 제외하고 나머지 텐서 펼치기(1,3136)
fc = nn.Linear(512, 10) # input_dim = 128*2*2=512, output_dim = 10
out = fc(out)

torch.manual_seed(777)# 랜덤 시드 고정

if device == 'cuda':
    torch.cuda.manual_seed_all(777)# GPU 사용 가능일 경우 랜덤 시드 고정

learning_rate = 0.001#파라미터 설정
training_epochs = 15
batch_size = 100
mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5  # 드롭아웃 확률

        # L1: 첫 번째 합성곱층 (Conv Layer)
        # 입력 이미지 형태: (?, 28, 28, 1)
        # Conv2d: 출력 채널 32개, 커널 크기 3x3, 스트라이드 1, 패딩 1
        # ReLU: 활성화 함수
        # MaxPool2d: 커널 크기 2x2, 스트라이드 2로 다운샘플링 -> 출력 형태: (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L2: 두 번째 합성곱층 (Conv Layer)
        # 입력 이미지 형태: (?, 14, 14, 32)
        # Conv2d: 출력 채널 64개, 커널 크기 3x3, 스트라이드 1, 패딩 1
        # ReLU: 활성화 함수
        # MaxPool2d: 커널 크기 2x2, 스트라이드 2로 다운샘플링 -> 출력 형태: (?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L3: 세 번째 합성곱층 (Conv Layer)
        # 입력 이미지 형태: (?, 7, 7, 64)
        # Conv2d: 출력 채널 128개, 커널 크기 3x3, 스트라이드 1, 패딩 1
        # ReLU: 활성화 함수
        # MaxPool2d: 커널 크기 2x2, 스트라이드 2, 패딩 1로 다운샘플링 -> 출력 형태: (?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4: 첫 번째 선형층 (Fully Connected Layer)
        # 입력 노드 수: 4x4x128, 출력 노드 수: 625
        # ReLU: 활성화 함수
        # Dropout: 드롭아웃으로 과적합 방지, p=0.5
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)  # 가중치 초기화
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))

        # L5: 최종 선형층 (Fully Connected Layer)
        # 입력 노드 수: 625, 출력 노드 수: 10 (클래스 개수)
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)  # 가중치 초기화

    def forward(self, x):
        out = self.layer1(x)  # 첫 번째 합성곱층 통과
        out = self.layer2(out)  # 두 번째 합성곱층 통과
        out = self.layer3(out)  # 세 번째 합성곱층 통과
        out = out.view(out.size(0), -1)  # 선형층에 입력하기 위해 텐서를 Flatten
        out = self.layer4(out)  # 첫 번째 선형층 통과
        out = self.fc2(out)  # 최종 선형층 통과
        return out  # 최종 출력 반환

# CNN 모델 정의
model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

   # print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost)) 에러값 제대로 나오는지 확인

# 학습을 진행하지 않을 것이므로 torch.no_grad()
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

