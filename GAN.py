import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 1. MNIST 1,5,6 필터링 데이터 준비 =====
transform = transforms.ToTensor()

mnist_train = datasets.MNIST(root='./MNIST_data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./MNIST_data', train=False, transform=transform, download=True)

def filter_classes(dataset, classes=[1,5,6]):
    imgs = []
    labels = []
    for img, label in dataset:
        if label in classes:
            imgs.append(img.numpy())
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

train_imgs_tensor = torch.from_numpy(train_imgs).float().unsqueeze(1).to(device)  # (N,1,28,28)
test_imgs_tensor = torch.from_numpy(test_imgs).float().unsqueeze(1).to(device)
train_labels_tensor = torch.from_numpy(train_labels).long().to(device)
test_labels_tensor = torch.from_numpy(test_labels).long().to(device)

# ===== 2. Fully Connected Autoencoder =====
class FullyConnectedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 300),
            nn.Tanh(),
            nn.Linear(300, 500),
            nn.Tanh(),
            nn.Linear(500, 28*28),
            nn.Sigmoid(),  # 픽셀 값 0~1로 맞춤
        )
    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out
    def encode(self, x):
        return self.encoder(x)
    def decode(self, z):
        return self.decoder(z)

fc_autoencoder = FullyConnectedAutoencoder().to(device)
fc_criterion = nn.MSELoss()
fc_optimizer = optim.Adam(fc_autoencoder.parameters(), lr=0.0001)

# ===== 3. Convolutional Autoencoder =====
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28x1 -> 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1),# 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # 14x14 -> 7x7
            nn.Conv2d(64, 2, 7),             # 7x7x2 latent
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 64, 7),    # 1x1x2 -> 7x7x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid(),
        )
    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out
    def encode(self, x):
        return self.encoder(x)
    def decode(self, z):
        return self.decoder(z)

conv_autoencoder = ConvAutoencoder().to(device)
conv_criterion = nn.MSELoss()
conv_optimizer = optim.Adam(conv_autoencoder.parameters(), lr=0.0001)

# ===== 4. Simple GAN 모델 =====
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),
        )
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x.view(-1, 28*28))

generator = Generator().to(device)
discriminator = Discriminator().to(device)
gan_criterion = nn.BCELoss()
gan_lr = 0.0002
gen_optimizer = optim.Adam(generator.parameters(), lr=gan_lr)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=gan_lr)

# ===== 5. 학습/평가 함수 샘플 =====

def train_autoencoder(model, optimizer, criterion, data_tensor, n_epoch=20, batch_size=50):
    model.train()
    n_data = data_tensor.size(0)
    for epoch in range(n_epoch):
        perm = torch.randperm(n_data)
        epoch_loss = 0
        for i in range(0, n_data, batch_size):
            indices = perm[i:i+batch_size]
            batch = data_tensor[indices].to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epoch}, Loss: {epoch_loss/(n_data//batch_size):.6f}")

def train_gan(gen, disc, gen_opt, disc_opt, criterion, dataset, n_epoch=50, batch_size=100):
    gen.train()
    disc.train()
    n_data = len(dataset)
    for epoch in range(n_epoch):
        perm = torch.randperm(n_data)
        for i in range(0, n_data, batch_size):
            indices = perm[i:i+batch_size]
            real_imgs = torch.stack([dataset[j][0] for j in indices]).to(device)
            real_labels = torch.ones(real_imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(real_imgs.size(0), 1).to(device)
            noise = torch.randn(real_imgs.size(0), 100).to(device)

            # Train Discriminator
            disc_opt.zero_grad()
            outputs_real = disc(real_imgs)
            loss_real = criterion(outputs_real, real_labels)
            fake_imgs = gen(noise)
            outputs_fake = disc(fake_imgs.detach())
            loss_fake = criterion(outputs_fake, fake_labels)
            disc_loss = loss_real + loss_fake
            disc_loss.backward()
            disc_opt.step()

            # Train Generator
            gen_opt.zero_grad()
            outputs_fake = disc(fake_imgs)
            gen_loss = criterion(outputs_fake, real_labels)
            gen_loss.backward()
            gen_opt.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epoch}, D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")

# ===== 6. 예시: Fully Connected Autoencoder 학습 =====
train_autoencoder(fc_autoencoder, fc_optimizer, fc_criterion, 
                  train_imgs_tensor.view(-1, 28*28), n_epoch=20, batch_size=50)

# ===== 7. 예시: Convolutional Autoencoder 학습 =====
train_autoencoder(conv_autoencoder, conv_optimizer, conv_criterion, 
                  train_imgs_tensor, n_epoch=20, batch_size=50)

# ===== 8. 예시: GAN 학습 =====
# 전체 MNIST 데이터셋에서 1,5,6 포함 모든 데이터로 GAN 학습하고 싶다면
gan_dataset = mnist_train  # 또는 필요시 필터링

train_gan(generator, discriminator, gen_optimizer, disc_optimizer, gan_criterion, gan_dataset, n_epoch=50, batch_size=100)
