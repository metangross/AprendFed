"""Modelo de energia por rodada (docs/custo_energia_rodada.pdf, seções 1 e 7).

    E = A·f(p)·η·L + C·s(p) + D

Por enquanto só existe a largura p = 1, então f = s = 1. Os coeficientes saem
de grandezas físicas, na forma potência × tempo do FedLoad:

    computação:   P_cmp · t_amostra / v    por amostra treinada
    comunicação:  P_rádio · bits / vazão   ao baixar e ao enviar o modelo
    custo fixo:   D                        uma vez por rodada

P_cmp = P_ref × power-factor, com P_ref e t_amostra medidos uma única vez por
fedavg_app/profile_energy.py; v é o compute-speed do cliente. O tempo vem do
perfil, não do sim_time medido, então a energia só depende do que o cliente
processa e as runs continuam reproduzíveis.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
BATTERY_EMPTY_POLICIES = ("drop", "partial")
# Linha de base descontada da energia medida no perfil (profile_energy.py):
# chave do perfil com P_ref e com a potência da base, e descrição para o output.
BASELINES = {
    "idle": ("train_power_w", "idle_power_w",
             "repouso (P8): o treino paga o cálculo e o clock alto"),
    "awake": ("train_power_awake_w", "awake_idle_power_w",
              "GPU acordada (P1): o treino paga só o cálculo"),
}


def profile_path(model: str, image_size: int, batch_size: int) -> Path:
    """Arquivo de perfil de uma configuração (o tempo por amostra depende das três)."""
    return PROFILES_DIR / f"{model}-i{image_size}-b{batch_size}.json"


@lru_cache(maxsize=4)
def _read_profile(model: str, image_size: int, batch_size: int) -> dict:
    path = profile_path(model, image_size, batch_size)
    if not path.exists():
        raise FileNotFoundError(
            f"Perfil de energia {path.name} não encontrado. Rode "
            "`docker compose run --rm profiler` (com os outros containers "
            "parados) para a mesma combinação de model, image-size e batch-size."
        )
    return json.loads(path.read_text())


def load_profile(run_config: dict) -> dict:
    """Perfil medido para o model/image-size/batch-size do run."""
    rc = run_config
    return _read_profile(str(rc["model"]), int(rc["image-size"]), int(rc["batch-size"]))


def energy_summary(run_config: dict) -> dict:
    """Escolhas e valores de referência do modelo de energia, para o metrics.json.

    `energy_per_sample_j` é o de referência (power-factor = 1, compute-speed = 1).
    """
    rc = run_config
    prof = load_profile(rc)
    width = prof["widths"]["1.0"]
    power_key, base_key, description = BASELINES[str(rc["energy-baseline"])]
    return {
        "enabled": True,
        "baseline": str(rc["energy-baseline"]),
        "baseline_description": description,
        "baseline_power_w": prof[base_key],
        "profile": profile_path(str(rc["model"]), int(rc["image-size"]),
                                int(rc["batch-size"])).name,
        "profile_gpu": prof["gpu"],
        "profile_date": prof["date"],
        "time_per_sample_s": width["time_per_sample_s"],
        "train_power_w": width[power_key],
        "energy_per_sample_j": width[power_key] * width["time_per_sample_s"],
        "fixed_j": float(rc["energy-fixed-j"]),
        "battery_empty_policy": str(rc["battery-empty-policy"]),
    }


@dataclass(frozen=True)
class RoundEnergy:
    """Custos de uma rodada para um cliente, em joules."""

    per_sample: float  # computação por amostra treinada
    download: float
    upload: float
    fixed: float
    bw_down_mbps: float
    bw_up_mbps: float


def round_energy(
    run_config: dict, node_config: dict, num_params: int, server_round: int
) -> RoundEnergy:
    """Coeficientes da rodada. A vazão de cada sentido é sorteada em
    U(1, net-mbps) Mbps, com seed (seed, rodada, partição), como no FedLoad."""
    rc, nc = run_config, node_config
    prof = load_profile(rc)["widths"]["1.0"]
    power_key = BASELINES[str(rc["energy-baseline"])][0]
    p_cmp = float(prof[power_key]) * float(nc.get("power-factor", 1.0))
    speed = float(nc.get("compute-speed", 1.0))
    per_sample = p_cmp * float(prof["time_per_sample_s"]) / speed

    rng = np.random.default_rng(
        [int(rc["seed"]), server_round, int(nc["partition-id"]), 1]
    )
    tier = float(nc.get("net-mbps", 25.0))
    bw_down, bw_up = rng.uniform(1.0, max(tier, 1.0), size=2)
    bits = 32 * num_params  # float32
    radio = float(nc.get("radio-w", 1.8))
    return RoundEnergy(
        per_sample=per_sample,
        download=radio * bits / (bw_down * 1e6),
        upload=radio * bits / (bw_up * 1e6),
        fixed=float(rc["energy-fixed-j"]),
        bw_down_mbps=float(bw_down),
        bw_up_mbps=float(bw_up),
    )
