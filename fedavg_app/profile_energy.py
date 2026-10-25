"""Perfilamento de energia do treino local na GPU (seção 6.2 de
docs/custo_energia_rodada.pdf).

Uso (com os outros containers parados, pois o NVML mede a placa inteira):
    docker compose run --rm profiler
    docker compose run --rm profiler python -m fedavg_app.profile_energy --repeats 10
    docker compose run --rm profiler python -m fedavg_app.profile_energy --widths 0.75,0.5,0.25

Mede a potência ociosa e, depois, o treino com o mesmo `train()` dos clientes.
Desconta ociosa × tempo e grava em fedavg_app/profiles/ a energia e o tempo por
amostra. Os clientes leem esse arquivo, então durante os experimentos a energia
é calculada (não medida) e as runs continuam reproduzíveis.

Com --widths, mede cada largura do submodelo (HeteroFL) e acrescenta ao perfil
que já existe, sem mudar as larguras que não foram medidas de novo: remedir a
1,0 mudaria a energia das runs já feitas.
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

from fedavg_app.energy import PROFILES_DIR, parse_widths, profile_path, width_key
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

    def wait_idle(self, timeout: float = 90.0) -> None:
        """Espera a GPU voltar ao repouso (P8) depois de uma carga."""
        deadline = time.monotonic() + timeout
        while self.pstate() < 8 and time.monotonic() < deadline:
            time.sleep(1.0)

    def other_processes(self) -> int:
        """Processos na GPU além deste (o NVML não separa a energia de cada um)."""
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        return max(len(procs) - 1, 0)


def profile_width(args, meter: EnergyMeter, loader: DataLoader, device: torch.device,
                  width: float, p_idle: float, p_awake: float) -> dict:
    """Mede o treino do (sub)modelo de largura `width`: entrada de `widths` no perfil."""
    net = get_model(args.model, image_size=args.image_size, width=width)

    def run(epochs: int) -> dict:
        return train(net, loader, epochs, args.lr, args.momentum, device)

    # Aquecimento: inicializa o cuDNN para estas formas e estima quantas épocas dão
    # `--seconds`. Depois, como na 1ª medição do perfil, as repetições partem do repouso.
    warm = run(1)
    epochs = max(1, math.ceil(args.seconds / warm["compute_time_s"]))
    meter.wait_idle()
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
    return {
        "time_per_sample_s": t_sample,
        "time_per_sample_std_s": t_sample_std,
        "energy_per_sample_j": e_sample,
        "energy_per_sample_std_j": e_sample_std,
        # P_ref (W) = energia acima do repouso / tempo; é o que os clientes usam.
        "train_power_w": e_sample / t_sample,
        # Alternativa: só o que o cálculo gasta acima da GPU acordada.
        "energy_per_sample_awake_j": e_awake,
        "train_power_awake_w": e_awake / t_sample,
        "num_params": sum(t.numel() for t in net.parameters()),
        "idle_power_w": p_idle,
        "awake_idle_power_w": p_awake,
        "epochs_per_repeat": epochs,
        "date": datetime.now().isoformat(timespec="seconds"),
        "repeats": repeats,
    }


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
    parser.add_argument("--widths", default="1.0",
                        help='larguras a medir, ex.: "0.75,0.5,0.25" (padrão: só a 1,0)')
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    meter = EnergyMeter()
    if n := meter.other_processes():
        print(f"AVISO: {n} outro(s) processo(s) usando a GPU; a medição vai misturá-los.")

    x, y = _load_split(args.data_dir, True)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)
    widths = parse_widths(args.widths)
    net = get_model(args.model, image_size=args.image_size)
    n_params = sum(t.numel() for t in net.parameters())
    # Aquecimento antes das linhas de base: a "acordada" é a GPU logo após a carga.
    train(net, loader, 1, args.lr, args.momentum, device)

    # Duas linhas de base. Logo após a carga a GPU fica ~15 s em alto clock (P1)
    # sem calcular ("acordada"); depois cai para o repouso (P8). Descontar o
    # repouso cobra do treino também o custo de manter o clock alto.
    def idle_power(seconds: float) -> float:
        _, e, t = meter.measure(lambda: time.sleep(seconds))
        return e / t

    p_awake = idle_power(min(args.idle_seconds, 5.0))
    meter.wait_idle()
    p_idle = idle_power(args.idle_seconds)
    print(
        f"Repouso: {p_idle:.2f} W (P{meter.pstate()}), acordada: {p_awake:.2f} W "
        f"({meter.method})"
    )
    measured = {width_key(w): profile_width(args, meter, loader, device, w, p_idle, p_awake)
                for w in widths}
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
        "epochs_per_repeat": measured[max(measured, key=float)]["epochs_per_repeat"],
        # Uma entrada por largura do submodelo (HeteroFL).
        "widths": measured,
    }

    out = args.output or profile_path(args.model, args.image_size, args.batch_size)
    if out.exists():
        # Acrescenta ao perfil: as larguras não medidas agora (e o cabeçalho, que
        # descreve a medição da 1,0) continuam como estavam.
        old = json.loads(out.read_text())
        same = all(old[k] == result[k] for k in ("model", "image_size", "batch_size"))
        if not same:
            raise SystemExit(f"{out} é de outra configuração; use --output")
        if "1.0" not in measured:
            result = old | {"widths": old["widths"] | measured}
        else:
            result["widths"] = old["widths"] | measured
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if args.output is None:
        # O container roda como root: devolve o arquivo ao dono do repositório.
        owner = PROFILES_DIR.parent.stat()
        for p in (PROFILES_DIR, out):
            os.chown(p, owner.st_uid, owner.st_gid)
    for w, m in measured.items():
        print(f"largura {w}: P_ref = {m['train_power_w']:.1f} W, t_amostra = "
              f"{1e6 * m['time_per_sample_s']:.2f} µs (±{1e6 * m['time_per_sample_std_s']:.2f})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
