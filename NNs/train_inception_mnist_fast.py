# Faster variant of train_inception_mnist.py: the full 3-epoch/60k-image
# run took an unusually long time on this shared CPU node (30+ min with
# no sign of finishing), so this trains on a 10000-image subset for 1
# epoch instead -- still real MNIST data, just less of it, to get a real
# (if less accurate) checkpoint without an open-ended wait.
import struct
import numpy as np
import torch
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
    torch.set_num_threads(4)
    train_x = read_idx_images(f"{MNIST_DIR}/train-images-idx3-ubyte")[:10000]
    train_y = read_idx_labels(f"{MNIST_DIR}/train-labels-idx1-ubyte")[:10000]
    test_x = read_idx_images(f"{MNIST_DIR}/t10k-images-idx3-ubyte")[:2000]
    test_y = read_idx_labels(f"{MNIST_DIR}/t10k-labels-idx1-ubyte")[:2000]
    print(f"train: {train_x.shape}, test: {test_x.shape}", flush=True)

    train_x_t = torch.from_numpy(train_x)
    train_y_t = torch.from_numpy(train_y)
    test_x_t = torch.from_numpy(test_x)
    test_y_t = torch.from_numpy(test_y)

    model = InceptionMNIST()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 128
    n = train_x_t.shape[0]

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
        if start % (batch_size * 10) == 0:
            print(f"  batch {start}/{n}", flush=True)
    print(f"train loss {total_loss / n:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred = model(test_x_t).argmax(dim=1)
        acc = (pred == test_y_t).float().mean().item()
    print(f"test accuracy: {acc:.4f}", flush=True)

    torch.save(model.state_dict(), "NNs/inception_mnist.pth")
    print("saved NNs/inception_mnist.pth", flush=True)


if __name__ == "__main__":
    main()
