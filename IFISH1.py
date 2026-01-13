import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A) Generator and Discriminator
img_dim = 28 * 28       # example image size 28x28
z_dim = 100             # noise dimension
batch_size = 100

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, img_dim),
            nn.Tanh()           # outputs in [-1, 1]
        )

    def forward(self, noise):
        x = self.net(noise)
        return x.view(-1, 1, 28, 28)  # fake_images

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()        # probability real/fake
        )

    def forward(self, images):
        x = images.view(images.size(0), -1)
        return self.net(x)      # real_or_fake

# C) Dummy real fish images (replace with real dataloader)
real_images = torch.randn(batch_size, 1, 28, 28, device=device)  # placeholder

# D) Initialize networks
generator = Generator(z_dim, img_dim).to(device)
discriminator = Discriminator(img_dim).to(device)

# E) Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

# F) Training loop
for epoch in range(100):
    # -------- Train Discriminator --------
    noise = torch.randn(batch_size, z_dim, device=device)
    fake_images = generator(noise).detach()

    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    # Real images
    preds_real = discriminator(real_images)
    loss_real = criterion(preds_real, real_labels)

    # Fake images
    preds_fake = discriminator(fake_images)
    loss_fake = criterion(preds_fake, fake_labels)

    loss_D = loss_real + loss_fake

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    # -------- Train Generator --------
    noise = torch.randn(batch_size, z_dim, device=device)
    fake_images = generator(noise)
    preds_fake_for_G = discriminator(fake_images)
    loss_G = criterion(preds_fake_for_G, real_labels)  # want fake → real

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

# G) Generate new fish images
with torch.no_grad():
    new_fish_images = generator(torch.randn(batch_size, z_dim, device=device))

# Example: save one generated image
img = new_fish_images[0].cpu().squeeze().numpy()
plt.imshow((img + 1) / 2, cmap="gray")   # scale from [-1,1] to [0,1]
plt.axis("off")
plt.savefig("generated_fish.png")
