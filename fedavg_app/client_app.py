"""ClientApp: treinamento e avaliação locais na GPU."""

import json
import math
import time
from pathlib import Path

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Error,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from fedavg_app.energy import round_energy
from fedavg_app.task import get_device, get_model, load_partition, set_seed, test
from fedavg_app.task import train as train_fn

app = ClientApp()

DEADLINE_EXCEEDED = 100  # código de erro enviado ao servidor na política "drop"
BATTERY_DEPLETED = 101  # bateria acabou, ou não sobra o bastante para treinar


def _battery(context: Context) -> float | None:
    """Nível da bateria (J). O context.state persiste entre as rodadas do run.

    None = cliente sem `battery-init` no --node-config: a energia é só contabilizada.
    """
    if "battery" not in context.state:
        init = context.node_config.get("battery-init")
        if init is None:
            return None
        context.state["battery"] = MetricRecord({"level": float(init)})
    return float(context.state["battery"]["level"])


def _set_battery(context: Context, level: float) -> None:
    context.state["battery"] = MetricRecord({"level": level})


def _no_battery(msg: Message, reason: str) -> Message:
    return Message(Error(code=BATTERY_DEPLETED, reason=reason), reply_to=msg)


def _setup(msg: Message, context: Context):
    """Instancia o modelo com os pesos globais e carrega a partição local."""
    rc, nc = context.run_config, context.node_config
    # Embaralhamento dos batches: reproduzível, mas distinto por rodada e cliente.
    set_seed(
        int(rc["seed"]),
        int(msg.content["config"]["server-round"]),
        int(nc["partition-id"]),
    )
    # Os pesos vêm do servidor; aqui só a arquitetura importa (pretrained=False).
    net = get_model(str(rc["model"]), image_size=int(rc["image-size"]))
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainloader, valloader = load_partition(
        partition_id=int(nc["partition-id"]),
        num_partitions=int(nc["num-partitions"]),
        data_dir=str(rc["data-dir"]),
        batch_size=int(rc["batch-size"]),
        method=str(rc["partitioning"]),
        alpha=float(rc["dirichlet-alpha"]),
        seed=int(rc["seed"]),
    )
    return net, trainloader, valloader


def _info(context: Context) -> ConfigRecord:
    """Identifica a partição para o servidor agregar as respostas em ordem fixa."""
    return ConfigRecord({"partition-id": int(context.node_config["partition-id"])})


def _log_metrics(msg: Message, context: Context, phase: str, metrics: dict) -> None:
    """Acrescenta uma linha em outputs/run-<id>-<data>/clients/partition-<p>.jsonl."""
    pid = int(context.node_config["partition-id"])
    cfg = msg.content["config"]
    run_date = str(cfg["run-date"])
    record = {
        "run_id": context.run_id,
        "run_date": run_date,
        "run_time": str(cfg["run-time"]),
        "round": int(cfg["server-round"]),
        "phase": phase,
        "partition_id": pid,
        "node_id": context.node_id,
        "compute_speed": float(context.node_config.get("compute-speed", 1.0)),
        "timestamp": time.time(),
        **metrics,
    }
    out = Path(str(context.run_config["output-dir"])) / f"run-{context.run_id}-{run_date}"
    out = out / "clients"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"partition-{pid}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


@app.train()
def train(msg: Message, context: Context) -> Message:
    rc, nc = context.run_config, context.node_config
    policy = str(rc["straggler-policy"])
    deadline = float(rc["round-deadline"]) if policy != "wait" else 0.0
    use_energy = bool(rc["energy"])
    battery = _battery(context) if use_energy else None
    if battery is not None and battery <= 0:
        # Já morreu numa rodada anterior: responde sem baixar nem treinar.
        _log_metrics(msg, context, "train", {"status": "dead", "battery_before": 0.0,
                                             "battery_after": 0.0, "energy_j": 0.0})
        return _no_battery(msg, "bateria vazia")

    device = get_device()
    net, trainloader, _ = _setup(msg, context)

    # Energia da rodada: custo fixo + download antes do treino, upload no fim.
    cost, energy_log, budget = None, {}, math.inf
    if use_energy:
        num_params = sum(t.numel() for t in net.parameters())
        cost = round_energy(rc, nc, num_params, int(msg.content["config"]["server-round"]))
        energy_log = {
            "energy_baseline": str(rc["energy-baseline"]),
            "power_factor": float(nc.get("power-factor", 1.0)),
            "net_mbps": float(nc.get("net-mbps", 25.0)),
            "bw_down_mbps": cost.bw_down_mbps,
            "bw_up_mbps": cost.bw_up_mbps,
            "battery_before": battery,
        }
    pre = cost.fixed + cost.download if cost else 0.0
    if battery is not None:
        if battery < pre:
            _set_battery(context, 0.0)
            _log_metrics(msg, context, "train", {"status": "depleted", **energy_log,
                                                 "battery_after": 0.0, "energy_j": battery})
            return _no_battery(msg, "bateria acabou ao baixar o modelo")
        # "partial" reserva o upload: para de treinar a tempo de enviar o modelo.
        # "drop" treina até a bateria acabar, como um FedAvg sem controle de energia.
        reserve = cost.upload if rc["battery-empty-policy"] == "partial" else 0.0
        budget = battery - pre - reserve

    stats = train_fn(
        net,
        trainloader,
        epochs=int(rc["local-epochs"]),
        lr=float(msg.content["config"]["lr"]),
        momentum=float(rc["momentum"]),
        device=device,
        compute_speed=float(nc.get("compute-speed", 1.0)),
        deadline=deadline,
        energy_per_sample=cost.per_sample if cost else 0.0,
        energy_budget=budget,
    )
    stop = stats["stop_reason"]
    seen = stats["examples_processed"]
    if stop == "deadline":
        status = "dropped" if policy == "drop" else "partial"
    elif stop == "battery" and seen == 0:
        status = "skipped"  # "partial" sem energia para um batch + upload
    elif stop == "battery":
        status = "depleted" if rc["battery-empty-policy"] == "drop" else "partial"
    else:
        status = "ok"

    spent = pre + stats["energy_compute_j"]
    uploads = status in ("ok", "partial")
    if cost:
        spent += cost.upload if uploads else 0.0
        energy_log |= {
            "energy_compute_j": stats["energy_compute_j"],
            "energy_comm_j": cost.download + (cost.upload if uploads else 0.0),
            "energy_fixed_j": cost.fixed,
        }
    if battery is not None:
        if status == "depleted" or spent > battery:
            # Morreu no meio do treino ou do upload: gastou o que restava.
            status, spent = "depleted", battery
        energy_log["battery_after"] = battery - spent
        _set_battery(context, battery - spent)
    if cost:
        energy_log["energy_j"] = spent

    _log_metrics(
        msg,
        context,
        "train",
        {"status": status, "num_examples": len(trainloader.dataset), **stats, **energy_log},
    )

    if status == "dropped":
        # Não cumpriu o prazo: o servidor conta como falha e ignora este cliente.
        reason = f"prazo de {deadline}s excedido (sim_time={stats['sim_time_s']:.2f}s)"
        return Message(Error(code=DEADLINE_EXCEEDED, reason=reason), reply_to=msg)
    if status == "depleted":
        return _no_battery(msg, "bateria acabou durante a rodada")
    if status == "skipped":
        return _no_battery(msg, f"bateria insuficiente para treinar ({battery - spent:.2f} J)")

    # Peso no FedAvg = exemplos efetivamente processados, então um cliente que
    # treinou parcialmente ("partial") contribui proporcionalmente menos.
    metrics = {"train_loss": stats["train_loss"], "num-examples": seen}
    if cost:
        metrics["energy-j"] = spent
    if battery is not None:
        metrics["battery"] = battery - spent
    arrays = ArrayRecord(net.state_dict())
    content = RecordDict(
        {"arrays": arrays, "metrics": MetricRecord(metrics), "info": _info(context)}
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    # A avaliação não gasta bateria (como no LeanFed), mas um cliente morto não avalia.
    battery = _battery(context) if bool(context.run_config["energy"]) else None
    if battery is not None and battery <= 0:
        return _no_battery(msg, "bateria vazia")
    device = get_device()
    net, _, valloader = _setup(msg, context)
    loss, acc = test(net, valloader, device)
    _log_metrics(
        msg,
        context,
        "evaluate",
        {"eval_loss": loss, "eval_acc": acc, "num_examples": len(valloader.dataset)},
    )
    metrics = MetricRecord(
        {"eval_loss": loss, "eval_acc": acc, "num-examples": len(valloader.dataset)}
    )
    content = RecordDict({"metrics": metrics, "info": _info(context)})
    return Message(content=content, reply_to=msg)
