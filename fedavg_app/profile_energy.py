"""Perfilamento de energia do treino local na GPU (seção 6.2 de
docs/custo_energia_rodada.pdf).

Uso (com os outros containers parados, pois o NVML mede a placa inteira):
    docker compose run --rm profiler
    docker compose run --rm profiler python -m fedavg_app.profile_energy --repeats 10

Mede a potência ociosa e, depois, o treino com o mesmo `train()` dos clientes.
Desconta ociosa × tempo e grava em fedavg_app/profiles/ a energia e o tempo por
amostra. Os clientes leem esse arquivo, então durante os experimentos a energia
é calculada (não medida) e as runs continuam reproduzíveis.
"""

import argparse
import json
import math
import os
import statistics
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path

import pynvml
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedavg_app.energy import PROFILES_DIR, profile_path
from fedavg_app.task import _load_split, get_device, get_model, set_seed, train


class EnergyMeter:
    """Energia acumulada da GPU em joules.

    Usa o contador do driver (nvmlDeviceGetTotalEnergyConsumption, GPUs Volta
    ou mais novas). Sem ele, integra nvmlDeviceGetPowerUsage numa thread.
    """

    def __init__(self, index: int = 0, period: float = 0.02) -> None:
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.period = period
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.method = "energy-counter"
        except pynvml.NVMLError:
            self.method = "power-sampling"

    def _power_w(self) -> float:
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def measure(self, fn):
        """Executa `fn()` e retorna (resultado, energia em J, duração em s)."""
        torch.cuda.synchronize()
        if self.method == "energy-counter":
            e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return out, (e1 - e0) / 1000.0, wall

        energy, stop = 0.0, threading.Event()

        def sample() -> None:
            nonlocal energy
            last_t, last_p = time.perf_counter(), self._power_w()
            while not stop.wait(self.period):
                t, p = time.perf_counter(), self._power_w()
                energy += 0.5 * (p + last_p) * (t - last_t)
                last_t, last_p = t, p

        th = threading.Thread(target=sample, daemon=True)
        t0 = time.perf_counter()
        th.start()
        out = fn()
        torch.cuda.synchronize()
        stop.set()
        th.join()
        return out, energy, time.perf_counter() - t0

    def pstate(self) -> int:
        return pynvml.nvmlDeviceGetPerformanceState(self.handle)

    def other_processes(self) -> int:
        """Processos na GPU além deste (o NVML não separa a energia de cada um)."""
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        return max(len(procs) - 1, 0)


def _defaults() -> dict:
    """Valores do [tool.flwr.app.config] do pyproject.toml, se disponível."""
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())["tool"]["flwr"]["app"]["config"]


def main() -> None:
    cfg = _defaults()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg.get("model", "cnn"))
    parser.add_argument("--image-size", type=int, default=cfg.get("image-size", 32))
    parser.add_argument("--batch-size", type=int, default=cfg.get("batch-size", 64))
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 0.01))
    parser.add_argument("--momentum", type=float, default=cfg.get("momentum", 0.9))
    parser.add_argument("--data-dir", default=cfg.get("data-dir", "/data"))
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument(
        "--seconds", type=float, default=10.0,
        help="duração mínima de cada repetição (o NVML atualiza a cada ~100 ms)",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    meter = EnergyMeter()
    if n := meter.other_processes():
        print(f"AVISO: {n} outro(s) processo(s) usando a GPU; a medição vai misturá-los.")

    x, y = _load_split(args.data_dir, True)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)
    net = get_model(args.model, image_size=args.image_size)
    n_params = sum(t.numel() for t in net.parameters())

    def run(epochs: int) -> dict:
        return train(net, loader, epochs, args.lr, args.momentum, device)

    # Aquecimento: inicializa CUDA/cuDNN e estima quantas épocas dão `--seconds`.
    warm = run(1)
    epochs = max(1, math.ceil(args.seconds / warm["compute_time_s"]))

    # Duas linhas de base. Logo após a carga a GPU fica ~15 s em alto clock (P1)
    # sem calcular ("acordada"); depois cai para o repouso (P8). Descontar o
    # repouso cobra do treino também o custo de manter o clock alto.
    def idle_power(seconds: float) -> float:
        _, e, t = meter.measure(lambda: time.sleep(seconds))
        return e / t

    p_awake = idle_power(min(args.idle_seconds, 5.0))
    deadline = time.monotonic() + 90.0
    while meter.pstate() < 8 and time.monotonic() < deadline:
        time.sleep(1.0)
    p_idle = idle_power(args.idle_seconds)
    print(
        f"Repouso: {p_idle:.2f} W (P{meter.pstate()}), acordada: {p_awake:.2f} W "
        f"({meter.method})"
    )

    repeats = []
    for r in range(args.repeats):
        stats, energy, wall = meter.measure(lambda: run(epochs))
        samples = stats["examples_processed"]
        rep = {
            "samples": samples,
            "wall_s": wall,
            "compute_s": stats["compute_time_s"],
            "energy_j": energy,
            # Tempo por amostra = o mesmo compute_time que o cliente usa no sim_time.
            "time_per_sample_s": stats["compute_time_s"] / samples,
            "energy_per_sample_j": (energy - p_idle * wall) / samples,
            "energy_per_sample_awake_j": (energy - p_awake * wall) / samples,
            "avg_power_w": energy / wall,
        }
        repeats.append(rep)
        print(
            f"[{r + 1}/{args.repeats}] {samples} amostras em {wall:.1f} s, "
            f"{energy:.1f} J ({rep['avg_power_w']:.1f} W), "
            f"{1e6 * rep['energy_per_sample_j']:.1f} µJ/amostra acima do repouso"
        )

    def mean_std(key: str) -> tuple[float, float]:
        vals = [r[key] for r in repeats]
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    t_sample, t_sample_std = mean_std("time_per_sample_s")
    e_sample, e_sample_std = mean_std("energy_per_sample_j")
    e_awake, _ = mean_std("energy_per_sample_awake_j")
    handle = meter.handle
    result = {
        "model": args.model,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_params": n_params,
        "gpu": pynvml.nvmlDeviceGetName(handle),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "torch": torch.__version__,
        "method": meter.method,
        "date": datetime.now().isoformat(timespec="seconds"),
        "idle_power_w": p_idle,
        "awake_idle_power_w": p_awake,
        "epochs_per_repeat": epochs,
        # Uma entrada por largura do submodelo; por enquanto só o modelo completo.
        "widths": {
            "1.0": {
                "time_per_sample_s": t_sample,
                "time_per_sample_std_s": t_sample_std,
                "energy_per_sample_j": e_sample,
                "energy_per_sample_std_j": e_sample_std,
                # P_ref (W) = energia acima do repouso / tempo; é o que os clientes usam.
                "train_power_w": e_sample / t_sample,
                # Alternativa: só o que o cálculo gasta acima da GPU acordada.
                "energy_per_sample_awake_j": e_awake,
                "train_power_awake_w": e_awake / t_sample,
                "repeats": repeats,
            }
        },
    }

    out = args.output or profile_path(args.model, args.image_size, args.batch_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if args.output is None:
        # O container roda como root: devolve o arquivo ao dono do repositório.
        owner = PROFILES_DIR.parent.stat()
        for p in (PROFILES_DIR, out):
            os.chown(p, owner.st_uid, owner.st_gid)
    print(
        f"P_ref = {result['widths']['1.0']['train_power_w']:.1f} W, "
        f"t_amostra = {1e6 * t_sample:.2f} µs (±{1e6 * t_sample_std:.2f}) -> {out}"
    )


if __name__ == "__main__":
    main()
