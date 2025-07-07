import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 1. 데이터 준비 (MNIST 1,5,6 필터링)
transform = transforms.ToTensor()

mnist_train = datasets.MNIST(root='./MNIST_data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./MNIST_data', train=False, transform=transform, download=True)

def filter_classes(dataset, classes=[1,5,6]):
    imgs = []
    labels = []
    for img, label in dataset:
        if label in classes:
            imgs.append(img.view(-1).numpy())
            labels.append(label)
    imgs = np.stack(imgs)
    labels = np.array(labels)
    return imgs, labels

train_imgs, train_labels = filter_classes(mnist_train, [1,5,6])
test_imgs, test_labels = filter_classes(mnist_test, [1,5,6])

n_train = train_imgs.shape[0]
n_test = test_imgs.shape[0]

print(f'Number of training images: {n_train}')
print(f'Number of test images: {n_test}')

# 2. 모델 정의
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 2),   # latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 300),
            nn.Tanh(),
            nn.Linear(300, 500),
            nn.Tanh(),
            nn.Linear(500, 28*28),
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)

# 3. 손실 함수 및 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 4. 텐서 변환
train_imgs_tensor = torch.from_numpy(train_imgs).float().to(device)
test_imgs_tensor = torch.from_numpy(test_imgs).float().to(device)
train_labels_tensor = torch.from_numpy(train_labels).long().to(device)
test_labels_tensor = torch.from_numpy(test_labels).long().to(device)

# 5. 학습 루프
n_batch = 50
n_epoch = 20
n_iter = n_epoch * (n_train // n_batch)
n_prt = n_train // n_batch

cost_record_train = []
cost_record_test = []

model.train()
for i in range(n_iter):
    idx = np.random.randint(0, n_train, n_batch)
    batch = train_imgs_tensor[idx]

    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, batch)
    loss.backward()
    optimizer.step()

    if i % n_prt == 0:
        model.eval()
        with torch.no_grad():
            train_loss = loss.item()
            idx_test = np.random.randint(0, n_test, n_batch)
            test_batch = test_imgs_tensor[idx_test]
            test_output = model(test_batch)
            test_loss = criterion(test_output, test_batch).item()
            
            cost_record_train.append(train_loss)
            cost_record_test.append(test_loss)
            
            print(f"Epoch: {i//n_prt + 1}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        model.train()

# 6. 손실 그래프
plt.plot(np.arange(1, len(cost_record_train)+1)*n_prt, cost_record_train, label='Training')
plt.plot(np.arange(1, len(cost_record_test)+1)*n_prt, cost_record_test, label='Test')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 7. 테스트 이미지 재구성 시각화
model.eval()
idx = np.random.randint(0, n_test)
test_x = test_imgs_tensor[idx].unsqueeze(0)
with torch.no_grad():
    x_reconst = model(test_x)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(test_x.cpu().squeeze().view(28,28), cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(x_reconst.cpu().squeeze().view(28,28), cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

plt.show()

# 8. 잠재 공간(latent space) 시각화
idxs = np.random.choice(n_test, 500, replace=False)
vis_test_imgs = test_imgs_tensor[idxs]
vis_test_labels = test_labels_tensor[idxs].cpu().numpy()

with torch.no_grad():
    latent = model.encode(vis_test_imgs).cpu().numpy()

plt.figure(figsize=(6,6))
plt.scatter(latent[vis_test_labels == 1, 0], latent[vis_test_labels == 1, 1], label='1', alpha=0.5)
plt.scatter(latent[vis_test_labels == 5, 0], latent[vis_test_labels == 5, 1], label='5', alpha=0.5)
plt.scatter(latent[vis_test_labels == 6, 0], latent[vis_test_labels == 6, 1], label='6', alpha=0.5)
plt.title('Latent space')
plt.xlabel('z1')
plt.ylabel('z2')
plt.legend()
plt.axis('equal')
plt.show()

# 9. 새로운 latent 벡터로 이미지 생성
new_data = torch.tensor([[5., -10.]]).to(device)

with torch.no_grad():
    generated_img = model.decode(new_data).cpu().numpy()

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(latent[vis_test_labels == 1, 0], latent[vis_test_labels == 1, 1], label='1', alpha=0.5)
plt.scatter(latent[vis_test_labels == 5, 0], latent[vis_test_labels == 5, 1], label='5', alpha=0.5)
plt.scatter(latent[vis_test_labels == 6, 0], latent[vis_test_labels == 6, 1], label='6', alpha=0.5)
plt.scatter(new_data.cpu().numpy()[:,0], new_data.cpu().numpy()[:,1], c='k', marker='o', s=100, label='New data')
plt.title('Latent space')
plt.xlabel('z1')
plt.ylabel('z2')
plt.legend()
plt.axis('equal')

plt.subplot(1,2,2)
plt.imshow(generated_img.reshape(28,28), cmap='gray')
plt.title('Generated fake image')
plt.xticks([])
plt.yticks([])

plt.show()

# 10. 잠재 공간 그리드(Manifold) 시각화
nx, ny = 20, 20
x_values = np.linspace(-7, 15, nx)
y_values = np.linspace(-12, 5, ny)
canvas = np.empty((28*ny, 28*nx))

for i, yi in enumerate(y_values):
    for j, xi in enumerate(x_values):
        z = torch.tensor([[xi, yi]]).float().to(device)
        with torch.no_grad():
            reconst_ = model.decode(z).cpu().numpy()
        canvas[(ny - i - 1)*28:(ny - i)*28, j*28:(j+1)*28] = reconst_.reshape(28,28)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(latent[vis_test_labels == 1, 0], latent[vis_test_labels == 1, 1], label='1', alpha=0.5)
plt.scatter(latent[vis_test_labels == 5, 0], latent[vis_test_labels == 5, 1], label='5', alpha=0.5)
plt.scatter(latent[vis_test_labels == 6, 0], latent[vis_test_labels == 6, 1], label='6', alpha=0.5)
plt.scatter(new_data.cpu().numpy()[:,0], new_data.cpu().numpy()[:,1], c='k', marker='o', s=100, label='New da
