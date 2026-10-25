"""Modelo, dados (CIFAR-10 ou FEMNIST particionados) e laços de treino/avaliação."""

import math
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

DATASETS = ("cifar10", "femnist")
NUM_CLASSES = {"cifar10": 10, "femnist": 62}


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
    """CNN simples para imagens 3x32x32. `width` escala os canais e a fc1."""

    def __init__(self, num_classes: int = 10, width: float = 1.0) -> None:
        super().__init__()
        c1, c2, c3, h = (max(1, round(c * width)) for c in (32, 64, 128, 256))
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(c3 * 4 * 4, h)
        self.fc2 = nn.Linear(h, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Resize(nn.Module):
    """Adapta a entrada antes do modelo, na GPU: replica imagens em tons de cinza
    (FEMNIST) nos 3 canais e amplia para `size` x `size`."""

    def __init__(self, net: nn.Module, size: int) -> None:
        super().__init__()
        self.net, self.size = net, size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return self.net(x)


MODELS = ("cnn", "mobilenet_v3_small")


def get_model(name: str, pretrained: bool = False, image_size: int = 32,
              dataset: str = "cifar10", width: float = 1.0) -> nn.Module:
    """Cria o modelo. Servidor e clientes devem usar os mesmos argumentos
    (exceto `pretrained`, que só importa para os pesos iniciais no servidor).

    `width` < 1 dá o submodelo do HeteroFL: cada tensor dele é o prefixo do
    tensor correspondente no modelo completo (submodel_state).
    """
    num_classes = NUM_CLASSES[dataset]
    if pretrained and width != 1.0:
        raise ValueError("Pesos pré-treinados só existem para o modelo completo")
    if name == "cnn":
        if pretrained:
            raise ValueError("O modelo 'cnn' não tem pesos pré-treinados")
        net: nn.Module = Net(num_classes, width)
    elif name == "mobilenet_v3_small":
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        # O torchvision não aceita width_mult com pesos; o completo é o padrão (1,0).
        net = (mobilenet_v3_small(weights=weights) if width == 1.0
               else mobilenet_v3_small(width_mult=width))
        # Troca a última camada (1000 classes do ImageNet) pela do dataset.
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Modelo desconhecido: {name!r}. Opções: {MODELS}")
    # O FEMNIST (1x28x28) sempre passa pelo Resize; o CIFAR-10 só se for ampliado.
    return Resize(net, image_size) if image_size != 32 or dataset != "cifar10" else net


def submodel(name: str, image_size: int, dataset: str, width: float) -> nn.Module:
    """Submodelo sem tocar nas RNGs globais: os pesos aleatórios dele são trocados
    pelo recorte do modelo global, e o sorteio dos batches continua o mesmo
    em qualquer largura."""
    with torch.random.fork_rng(devices=[]):
        return get_model(name, image_size=image_size, dataset=dataset, width=width)


@lru_cache(maxsize=16)
def num_params(name: str, image_size: int, dataset: str, width: float) -> int:
    """Parâmetros do submodelo de largura `width` (o que ele envia e recebe)."""
    return sum(t.numel() for t in submodel(name, image_size, dataset, width).parameters())


def static_bn(net: nn.Module) -> None:
    """sBN do HeteroFL: a BatchNorm treina com as estatísticas do batch e não
    acumula running_mean/var, que voltam ao servidor como vieram. O modelo
    global é avaliado com as estatísticas recalculadas (recalibrate_bn)."""
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False


def add_scaler(net: nn.Module, width: float) -> None:
    """Scaler do HeteroFL: no treino, divide por `width` a entrada de cada
    BatchNorm do submodelo, como no código do artigo (`n1(scaler(x))`); camadas
    sem normalização depois, como o classificador, não são escaladas. Na
    avaliação não faz nada. Fica em hooks, fora do state_dict, então o recorte e
    a agregação não mudam."""
    def scale(module: nn.Module, inputs: tuple) -> tuple | None:
        return (inputs[0] / width, *inputs[1:]) if module.training else None

    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.register_forward_pre_hook(scale)


def submodel_state(full: dict[str, torch.Tensor], net: nn.Module) -> dict[str, torch.Tensor]:
    """Recorte do estado do modelo completo no formato de `net` (HeteroFL): o
    prefixo de cada tensor, em todas as dimensões."""
    shapes = {k: v.shape for k, v in net.state_dict().items()}
    return {k: v[tuple(slice(0, n) for n in shapes[k])] for k, v in full.items()}


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


FEMNIST_URL = "https://huggingface.co/datasets/flwrlabs/femnist/resolve/main/data/train-00000-of-00001.parquet"
# Escritores separados para o teste centralizado: sorteados com uma seed fixa,
# são os mesmos em todas as runs e nunca vão para um cliente.
FEMNIST_TEST_WRITERS = 50
FEMNIST_TEST_SEED = 0


def _femnist_path(data_dir: str, name: str) -> Path:
    return Path(data_dir) / f"femnist_{name}.npy"


def download_femnist(data_dir: str) -> None:
    """Baixa o FEMNIST (814 mil caracteres 28x28 de 3.597 escritores) e salva em
    .npy: imagens uint8, rótulos (62 classes) e o índice do escritor de cada uma.
    Em .npy, os clientes leem com mmap só os índices de que precisam."""
    import io
    import urllib.request

    import pyarrow.parquet as pq
    from PIL import Image

    if _femnist_path(data_dir, "writer").exists():
        return
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print(f"Baixando {FEMNIST_URL}", flush=True)
    with urllib.request.urlopen(FEMNIST_URL) as resp:
        table = pq.read_table(io.BytesIO(resp.read()))
    images = np.stack(
        [np.array(Image.open(io.BytesIO(img["bytes"])).convert("L"))
         for img in table.column("image").to_pylist()]
    )
    labels = np.array(table.column("character").to_pylist(), dtype=np.int64)
    _, writers = np.unique(np.array(table.column("writer_id").to_pylist()),
                           return_inverse=True)
    # O escritor por último: ele marca o download como completo.
    for name, arr in (("x", images), ("y", labels), ("writer", writers.astype(np.int32))):
        tmp = _femnist_path(data_dir, name + ".tmp")
        np.save(tmp, arr)
        tmp.rename(_femnist_path(data_dir, name))
    print(f"FEMNIST: {images.shape}, {writers.max() + 1} escritores", flush=True)


@lru_cache(maxsize=1)
def _femnist_arrays(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Imagens (mmap, uint8), rótulos e escritores do FEMNIST."""
    if not _femnist_path(data_dir, "writer").exists():
        raise FileNotFoundError(
            "FEMNIST não encontrado em " + data_dir + ". Rode "
            "`docker compose run --rm -v ./:/app dataset "
            "python -m fedavg_app.prepare_data /data femnist`."
        )
    return (np.load(_femnist_path(data_dir, "x"), mmap_mode="r"),
            np.load(_femnist_path(data_dir, "y")),
            np.load(_femnist_path(data_dir, "writer")))


def _femnist_tensors(data_dir: str, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Converte só as imagens `idx`: traço claro sobre fundo 0, em [-1, 1]."""
    x, y, _ = _femnist_arrays(data_dir)
    xs = 1.0 - torch.from_numpy(np.asarray(x[idx])).float().div(255.0)
    return (xs.unsqueeze(1) - 0.5) / 0.5, torch.from_numpy(y[idx])


def femnist_test_writers(num_writers: int) -> np.ndarray:
    return np.random.default_rng(FEMNIST_TEST_SEED).permutation(num_writers)[
        :FEMNIST_TEST_WRITERS]


@lru_cache(maxsize=2)
def femnist_partitions(data_dir: str, num_partitions: int, method: str, alpha: float,
                       seed: int, writers_per_client: int) -> list[np.ndarray]:
    """Divide o FEMNIST entre os clientes.

    A seed sorteia num_partitions × writers_per_client escritores (fora os do teste).
    "writer" dá a cada cliente os seus escritores inteiros (partição natural, por
    usuário); "iid" e "dirichlet" redistribuem as amostras desses mesmos escritores.
    """
    _, y, w = _femnist_arrays(data_dir)
    num_writers = int(w.max()) + 1
    test = femnist_test_writers(num_writers)
    pool = np.setdiff1d(np.arange(num_writers), test)
    need = num_partitions * writers_per_client
    if need > len(pool):
        raise ValueError(f"FEMNIST tem {len(pool)} escritores para os clientes; "
                         f"{num_partitions} × {writers_per_client} = {need}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=need, replace=False)
    if method == "writer":
        groups = chosen.reshape(num_partitions, writers_per_client)
        return [rng.permutation(np.flatnonzero(np.isin(w, g))) for g in groups]
    sample = np.flatnonzero(np.isin(w, chosen))
    parts = partition_indices(y[sample], num_partitions, method, alpha, seed)
    return [sample[p] for p in parts]


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
    dataset: str = "cifar10",
    writers_per_client: int = 12,
) -> tuple[DataLoader, DataLoader]:
    """Retorna (train, val) da partição local do cliente (split 80/20)."""
    if dataset == "femnist":
        idx = femnist_partitions(data_dir, num_partitions, method, alpha, seed,
                                 writers_per_client)[partition_id]
        x, y = _femnist_tensors(data_dir, idx)
        idx = np.arange(len(idx))
    else:
        x, y = _load_split(data_dir, True)
        idx = partition_indices(y.numpy(), num_partitions, method, alpha, seed)[partition_id]
    n_val = max(1, int(0.2 * len(idx)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    trainloader = DataLoader(
        TensorDataset(x[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(TensorDataset(x[val_idx], y[val_idx]), batch_size=512)
    return trainloader, valloader


def load_centralized_test(data_dir: str, batch_size: int = 512,
                          dataset: str = "cifar10") -> DataLoader:
    if dataset == "femnist":
        _, _, w = _femnist_arrays(data_dir)
        idx = np.flatnonzero(np.isin(w, femnist_test_writers(int(w.max()) + 1)))
        x, y = _femnist_tensors(data_dir, idx)
    else:
        x, y = _load_split(data_dir, False)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def load_calibration(data_dir: str, num_partitions: int, method: str, alpha: float,
                     seed: int, dataset: str = "cifar10", writers_per_client: int = 12,
                     size: int = 2048, batch_size: int = 64) -> DataLoader:
    """Amostra fixa (pela seed) dos dados dos clientes, para recalibrar a BatchNorm.

    No CIFAR-10 os clientes dividem o treino inteiro; no FEMNIST, as amostras dos
    escritores sorteados para eles. Não usa o teste.
    """
    if dataset == "femnist":
        pool = np.concatenate(femnist_partitions(data_dir, num_partitions, method, alpha,
                                                 seed, writers_per_client))
        idx = np.random.default_rng(seed).choice(pool, size=min(size, len(pool)),
                                                 replace=False)
        x, y = _femnist_tensors(data_dir, np.sort(idx))
    else:
        x, y = _load_split(data_dir, True)
        idx = np.random.default_rng(seed).choice(len(y), size=min(size, len(y)),
                                                 replace=False)
        x, y = x[idx], y[idx]
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
    energy_per_sample: float = 0.0,
    energy_budget: float = math.inf,
) -> dict:
    """Treino local com emulação de hardware heterogêneo.

    `compute_speed` < 1 emula um dispositivo mais lento: após cada batch o
    cliente dorme o tempo extra que um hardware `1/compute_speed` vezes mais
    lento levaria. `sim_time` é o tempo (emulado) que o cliente "gastou".
    Se `deadline` > 0, o treino para quando `sim_time` atinge o prazo.

    Cada batch custa `energy_per_sample` × tamanho do batch (J). O treino para
    antes do batch que ultrapassaria `energy_budget`; `stop_reason` diz se ele
    parou pelo prazo ("deadline") ou pela energia ("battery").
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
    compute_time, sim_time, energy = 0.0, 0.0, 0.0
    stop_reason = ""
    for _ in range(epochs):
        for xb, yb in loader:
            cost = energy_per_sample * len(yb)
            if energy + cost > energy_budget:
                stop_reason = "battery"
                break
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
            energy += cost
            seen += len(yb)
            steps += 1
            if deadline > 0 and sim_time >= deadline:
                stop_reason = "deadline"
                break
        if stop_reason:
            break
    return {
        "train_loss": total_loss / max(seen, 1),
        "examples_processed": seen,
        "steps": steps,
        "compute_time_s": compute_time,
        "sim_time_s": sim_time,
        "energy_compute_j": energy,
        "stop_reason": stop_reason,
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


@torch.no_grad()
def recalibrate_bn(net: nn.Module, loader: DataLoader, device: torch.device) -> None:
    """Refaz running_mean/var da BatchNorm com os pesos atuais (os pesos não mudam).

    A média dos buffers feita pela FedAvg vem dos modelos locais e ignora a
    dispersão entre clientes; com dados não-IID, às vezes derruba a acurácia do
    modelo global para perto do acaso numa rodada.
    """
    net.to(device).eval()  # só a BN em modo de treino: o dropout fica desligado
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # média cumulativa sobre os batches, não móvel
        m.train()
    for xb, _ in loader:
        net(xb.to(device, non_blocking=True))
    net.eval()
