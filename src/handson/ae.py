#%%
import os
from pathlib import Path

BASE = Path("/tmp") / "hf_cache"
BASE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(BASE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(BASE / "hub")
os.environ["HF_DATASETS_CACHE"] = str(BASE / "datasets")
os.environ["XDG_CACHE_HOME"] = str(BASE / "xdg")

#%%
from datasets import load_dataset
from genaibook.core import show_images
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'gray_r'
mnist = load_dataset('mnist', cache_dir="./hf_cache")
# %%
mnist
#%%

show_images(mnist['train']['image'][:4])
# %%
from torchvision import transforms

def mnist_to_tensor(samples):
    if 'image' in samples:  # 'image'キーが存在する場合のみ変換
        t = transforms.ToTensor()
        samples['image'] = [t(image) for image in samples['image']]
    return samples

mnist = mnist.with_transform(mnist_to_tensor)
mnist['train'] = mnist['train'].shuffle(seed=1337)
x = mnist['train']['image'][0]
x.min(), x.max()




# %%
show_images(mnist['train']['image'][:4])
# %%
y = mnist['train']['label'][:4]
y
# %%
from torch.utils.data import DataLoader
bs = 4
train_dataloader = DataLoader(mnist['train']['image'], batch_size=bs)

# %% encoder
from torch import nn

def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = conv_block(in_channels, 128)
        self.conv2 = conv_block(128, 256)
        self.conv3 = conv_block(256, 512)
        self.conv4 = conv_block(512, 1024)
        self.linear = nn.Linear(1024, 16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.linear(x.flatten(start_dim=1))
        return x
    
#%% test
mnist['train']['image'][0].shape
in_channels = 1
x = mnist['train']['image'][0][None, :]
encoder = Encoder(in_channels).eval()
encoded = encoder(x)
encoded.shape
# %%
encoded
# %%
batch = next(iter(train_dataloader))
encoded = Encoder(in_channels)(batch)
batch.shape, encoded.shape
# %% decoder
def conv_transpose_block(
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=2, 
        padding=1,
        output_padding=0,
        with_act=True):
    modules = [
        nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding),
    ]
    if with_act:
        modules += [
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
    return nn.Sequential(*modules)

class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.linear = nn.Linear(16, 1024 * 4 * 4)
        self.t_conv1 = conv_transpose_block(1024, 512)
        self.t_conv2 = conv_transpose_block(512, 256, output_padding=1)
        self.t_conv3 = conv_transpose_block(256, out_channels, output_padding=1)

    def forward(self, x):
        bs = x.shape[0]
        x = self.linear(x)
        x = x.reshape((bs, 1024, 4, 4))
        x = self.t_conv1(x)
        x = self.t_conv2(x)
        x = self.t_conv3(x)
        return x
    
#%% test decoder
decoded_batch = Decoder(out_channels=1)(encoded)
decoded_batch.shape
# %%
class AutoEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(in_channels)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encoder(x))
    
#%%
model = AutoEncoder(in_channels=1)

# %% 
import torchsummary
torchsummary.summary(model, (1, 28, 28), device='cpu')

# %% Training
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm.notebook import tqdm, trange

from genaibook.core import get_device

num_epochs = 2
lr = 1e-4

device = get_device()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)

losses = []  # 描画用に損失を保存するリスト
for _ in (progress := trange(num_epochs, desc="Training")):
    for _, batch in (
        inner := tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    ):
        batch = batch.to(device)

        # モデルに渡して再構成画像を取得する
        preds = model(batch)

        # 予測結果と元画像を比較する
        loss = F.mse_loss(preds, batch)

        # 損失を表示し、描画用に保存する
        inner.set_postfix(loss=f"{loss.cpu().item():.3f}")
        losses.append(loss.item())

        # 損失に基づいてモデルのパラメーターを更新する
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    progress.set_postfix(loss=f"{loss.cpu().item():.3f}", lr=f"{lr:.0e}")
# %%
import matplotlib.pyplot as plt

print(plt.style.available)
plt.style.use('seaborn-v0_8')
# %%
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("AutoEncoder – Training Loss Curve")
plt.show()

# %%
eval_bs = 16
eval_dataloader = DataLoader(mnist["test"]["image"], batch_size=eval_bs)
# %%
model.eval()
with torch.inference_mode():
    eval_batch = next(iter(eval_dataloader))
    predicted = model(eval_batch.to(device)).cpu()
# %%
batch_vs_preds = torch.cat((eval_batch, predicted))
show_images(batch_vs_preds, imsize=1, nrows=2)
# %%
def plot_activation_fn(fn, name):
    x = torch.linspace(-5, 5, 100)
    y = fn(x)
    plt.plot(x, y, label=name)
    plt.legend()


plt.title("Activation Functions")
plot_activation_fn(F.relu, "ReLU")
plot_activation_fn(F.sigmoid, "Sigmoid")
# %%
images_labels_dataloader = DataLoader(mnist["test"], batch_size=512)

#%%
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()

        self.conv_layers = nn.Sequential(
            conv_block(in_channels, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 1024),
        )
        self.linear = nn.Linear(1024, latent_dims)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv_layers(x)
        x = self.linear(x.reshape(bs, -1))
        return x
# %%
class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dims):
        super().__init__()

        self.linear = nn.Linear(latent_dims, 1024 * 4 * 4)
        self.t_conv_layers = nn.Sequential(
            conv_transpose_block(1024, 512),
            conv_transpose_block(512, 256, output_padding=1),
            conv_transpose_block(
                256, out_channels, output_padding=1, with_act=False
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs = x.shape[0]
        x = self.linear(x)
        x = x.reshape((bs, 1024, 4, 4))
        x = self.t_conv_layers(x)
        x = self.sigmoid(x)
        return x
#%%
class AutoEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dims)
        self.decoder = Decoder(in_channels, latent_dims)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))
# %%
def train(model, num_epochs=10, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)

    model.train()  # モデルを訓練モードに設定する
    losses = []
    for _ in (progress := trange(num_epochs, desc="Training")):
        for _, batch in (
            inner := tqdm(
                enumerate(train_dataloader), total=len(train_dataloader)
            )
        ):
            batch = batch.to(device)

            # モデルに渡して別の画像セットを取得する
            preds = model(batch)

            # 予測結果と元画像を比較する
            loss = F.mse_loss(preds, batch)

            # 損失を表示し、描画用に保存する
            inner.set_postfix(loss=f"{loss.cpu().item():.3f}")
            losses.append(loss.item())

            # 損失に基づいてモデルのパラメーターを更新する
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        progress.set_postfix(loss=f"{loss.cpu().item():.3f}", lr=f"{lr:.0e}")
    return losses
# %%
ae_model = AutoEncoder(in_channels=1, latent_dims=2)
ae_model.to(device)
# %%
losses = train(ae_model, num_epochs=2)
# %%
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve (two latent dimensions)")
plt.show()
# %%
ae_model.eval()
with torch.inference_mode():
    eval_batch = next(iter(eval_dataloader))
    predicted = ae_model(eval_batch.to(device)).cpu()
# %%
batch_vs_preds = torch.cat((eval_batch, predicted))
show_images(batch_vs_preds, imsize=1, nrows=2)
# %%
images_labels_dataloader = DataLoader(mnist["test"], batch_size=512)
# %%
import pandas as pd

df = pd.DataFrame(
    {
        "x": [],
        "y": [],
        "label": [],
    }
)

for batch in tqdm(
    iter(images_labels_dataloader), total=len(images_labels_dataloader)
):
    encoded = ae_model.encode(batch["image"].to(device)).cpu()
    new_items = {
        "x": [t.item() for t in encoded[:, 0]],
        "y": [t.item() for t in encoded[:, 1]],
        "label": batch["label"],
    }
    df = pd.concat([df, pd.DataFrame(new_items)], ignore_index=True)
# %%
plt.figure(figsize=(10, 8))

for label in range(10):
    points = df[df["label"] == label]
    plt.scatter(points["x"], points["y"], label=label, marker=".")

plt.legend();
N = 16  # 16個の点を生成する
z = torch.rand((N, 2)) * 8 - 4
#%%
x = -4.0
z = torch.tensor([[2.0, -6.0], [3.0, -6.0], [4.0, -6.0], [4.0, -4.0], [x, -6.0], [x, -5.0], [x, -4.0], [x, -3.0], [x, -2.0], [x, -1.0], [x, 0.0], [x, 1.0], [x, 2.0], [x, 3.0], [x, 4.0], [x, 5.0]])
# %%
plt.figure(figsize=(10, 8))

for label in range(10):
    points = df[df["label"] == label]
    plt.scatter(points["x"], points["y"], label=label, marker=".")

plt.scatter(z[:, 0], z[:, 1], label="z", marker="s", color="black")

# zの各点に番号を付ける
for i in range(len(z)):
    plt.annotate(str(i), (z[i, 0], z[i, 1]), 
                fontsize=8, ha='center', 
                xytext=(3, 3), textcoords='offset points')

plt.legend();
# %%
ae_decoded = ae_model.decode(z.to(device))
show_images(ae_decoded.cpu(), imsize=1, nrows=1, suptitle="AutoEncoder")
# %%
