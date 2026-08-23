# Trains InceptionMNIST on real MNIST data (using the already-cached raw
# idx files under alpha-beta-CROWN's dataset dir -- no torchvision needed,
# just a plain idx-format parser). Run with the host `taso_py` conda env.
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inception_mnist_model import InceptionMNIST

MNIST_DIR = "alpha-beta-CROWN/complete_verifier/datasets/MNIST/raw"


def read_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, rows, cols)
    return data.astype(np.float32) / 255.0


def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def main():
    torch.manual_seed(0)
    train_x = read_idx_images(f"{MNIST_DIR}/train-images-idx3-ubyte")
    train_y = read_idx_labels(f"{MNIST_DIR}/train-labels-idx1-ubyte")
    test_x = read_idx_images(f"{MNIST_DIR}/t10k-images-idx3-ubyte")
    test_y = read_idx_labels(f"{MNIST_DIR}/t10k-labels-idx1-ubyte")
    print(f"train: {train_x.shape}, test: {test_x.shape}")

    train_x_t = torch.from_numpy(train_x)
    train_y_t = torch.from_numpy(train_y)
    test_x_t = torch.from_numpy(test_x)
    test_y_t = torch.from_numpy(test_y)

    model = InceptionMNIST()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 128
    n = train_x_t.shape[0]
    epochs = 3

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = train_x_t[idx], train_y_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
        print(f"epoch {epoch}: train loss {total_loss / n:.4f}")

    model.eval()
    with torch.no_grad():
        correct = 0
        for start in range(0, test_x_t.shape[0], 1000):
            xb = test_x_t[start:start + 1000]
            yb = test_y_t[start:start + 1000]
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
        acc = correct / test_x_t.shape[0]
    print(f"test accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "NNs/inception_mnist.pth")
    print("saved NNs/inception_mnist.pth")


if __name__ == "__main__":
    main()
