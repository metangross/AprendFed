"""Modelo, dados (CIFAR-10 particionado) e laços de treino/avaliação."""

import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def set_seed(*keys: int) -> None:
    """Fixa as RNGs do Python, NumPy e PyTorch (CPU e GPU) a partir de `keys`.

    Os clientes passam (seed, rodada, partição): cada um embaralha seus batches
    de um jeito diferente a cada rodada, mas de forma reproduzível entre runs.
    """
    s = int(np.random.SeedSequence(list(keys)).generate_state(1)[0])
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponível no container. Verifique o NVIDIA Container Toolkit "
            "e a seção 'deploy.resources' do docker-compose.yml."
        )
    return torch.device("cuda")


class Net(nn.Module):
    """CNN simples para CIFAR-10."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Resize(nn.Module):
    """Amplia a entrada (32x32) antes do modelo; feito na GPU."""

    def __init__(self, net: nn.Module, size: int) -> None:
        super().__init__()
        self.net, self.size = net, size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return self.net(x)


MODELS = ("cnn", "mobilenet_v3_small")


def get_model(name: str, pretrained: bool = False, image_size: int = 32) -> nn.Module:
    """Cria o modelo. Servidor e clientes devem usar os mesmos argumentos
    (exceto `pretrained`, que só importa para os pesos iniciais no servidor)."""
    if name == "cnn":
        if pretrained:
            raise ValueError("O modelo 'cnn' não tem pesos pré-treinados")
        net: nn.Module = Net()
    elif name == "mobilenet_v3_small":
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_small(weights=weights)
        # Troca a última camada (1000 classes do ImageNet) por uma de 10 classes.
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, 10)
    else:
        raise ValueError(f"Modelo desconhecido: {name!r}. Opções: {MODELS}")
    return Resize(net, image_size) if image_size != 32 else net


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"


def _npz_path(data_dir: str, train: bool) -> Path:
    return Path(data_dir) / f"cifar10_{'train' if train else 'test'}.npz"


def download_cifar10(data_dir: str) -> None:
    """Baixa o CIFAR-10 (espelho do Hugging Face) e salva como .npz."""
    import io
    import urllib.request

    import pyarrow.parquet as pq
    from PIL import Image

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    for train in (True, False):
        out = _npz_path(data_dir, train)
        if out.exists():
            continue
        url = HF_URL.format(split="train" if train else "test")
        print(f"Baixando {url}", flush=True)
        with urllib.request.urlopen(url) as resp:
            table = pq.read_table(io.BytesIO(resp.read()))
        images = np.stack(
            [np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
             for img in table.column("img").to_pylist()]
        )
        labels = np.array(table.column("label").to_pylist(), dtype=np.int64)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, x=images, y=labels)
        tmp.rename(out)
        print(f"{out}: {images.shape}", flush=True)


@lru_cache(maxsize=2)
def _load_split(data_dir: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Carrega o CIFAR-10 já normalizado como tensores (N, 3, 32, 32)."""
    with np.load(_npz_path(data_dir, train)) as d:
        x = torch.from_numpy(d["x"]).permute(0, 3, 1, 2).float().div(255.0)
        y = torch.from_numpy(d["y"])
    x = (x - CIFAR_MEAN) / CIFAR_STD
    return x, y


def partition_indices(
    labels: np.ndarray, num_partitions: int, method: str, alpha: float, seed: int
) -> list[np.ndarray]:
    """Divide os índices do conjunto de treino entre os clientes.

    O particionamento é determinístico (depende só da seed), então cada cliente
    calcula sozinho a sua parte sem precisar de coordenação.
    """
    rng = np.random.default_rng(seed)
    if method == "iid":
        return np.array_split(rng.permutation(len(labels)), num_partitions)
    if method == "dirichlet":
        parts: list[list[int]] = [[] for _ in range(num_partitions)]
        for c in np.unique(labels):
            idx = rng.permutation(np.where(labels == c)[0])
            props = rng.dirichlet(np.full(num_partitions, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for p, chunk in enumerate(np.split(idx, cuts)):
                parts[p].extend(chunk.tolist())
        return [rng.permutation(np.array(p, dtype=int)) for p in parts]
    raise ValueError(f"Particionamento desconhecido: {method!r}")


def load_partition(
    partition_id: int,
    num_partitions: int,
    data_dir: str,
    batch_size: int,
    method: str = "iid",
    alpha: float = 0.5,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Retorna (train, val) da partição local do cliente (split 80/20)."""
    x, y = _load_split(data_dir, True)
    idx = partition_indices(y.numpy(), num_partitions, method, alpha, seed)[partition_id]
    n_val = max(1, int(0.2 * len(idx)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    trainloader = DataLoader(
        TensorDataset(x[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(TensorDataset(x[val_idx], y[val_idx]), batch_size=512)
    return trainloader, valloader


def load_centralized_test(data_dir: str, batch_size: int = 512) -> DataLoader:
    x, y = _load_split(data_dir, False)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


# --------------------------------------------------------------------------- #
# Treino / avaliação
# --------------------------------------------------------------------------- #
def train(
    net: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    momentum: float,
    device: torch.device,
    compute_speed: float = 1.0,
    deadline: float = 0.0,
) -> dict:
    """Treino local com emulação de hardware heterogêneo.

    `compute_speed` < 1 emula um dispositivo mais lento: após cada batch o
    cliente dorme o tempo extra que um hardware `1/compute_speed` vezes mais
    lento levaria. `sim_time` é o tempo (emulado) que o cliente "gastou".
    Se `deadline` > 0, o treino para quando `sim_time` atinge o prazo.
    """
    net.to(device).train()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    # Aquecimento (fora da medição): o 1º forward/backward inclui a inicialização
    # do CUDA/cuDNN, que seria multiplicada pela lentidão emulada. O estado é
    # restaurado para não alterar as estatísticas de BatchNorm.
    state = {k: v.clone() for k, v in net.state_dict().items()}
    xb, _ = next(iter(loader))
    net(xb.to(device)).sum().backward()
    optimizer.zero_grad()
    net.load_state_dict(state)
    torch.cuda.synchronize(device)

    total_loss, seen, steps = 0.0, 0, 0
    compute_time, sim_time = 0.0, 0.0
    completed = True
    for _ in range(epochs):
        for xb, yb in loader:
            t0 = time.perf_counter()
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)  # .item() sincroniza com a GPU
            dt = time.perf_counter() - t0
            if compute_speed < 1.0:
                time.sleep(dt * (1.0 / compute_speed - 1.0))
            compute_time += dt
            sim_time += dt / compute_speed
            seen += len(yb)
            steps += 1
            if deadline > 0 and sim_time >= deadline:
                completed = False
                break
        if not completed:
            break
    return {
        "train_loss": total_loss / max(seen, 1),
        "examples_processed": seen,
        "steps": steps,
        "compute_time_s": compute_time,
        "sim_time_s": sim_time,
        "completed": completed,
    }


@torch.no_grad()
def test(net: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    net.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        out = net(xb)
        loss += criterion(out, yb).item()
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)
    return loss / max(total, 1), correct / max(total, 1)
