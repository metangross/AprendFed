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
from torch import randperm
from torch.utils.data import DataLoader, Subset

from fedavg_app.energy import (
    choose_width,
    data_fraction,
    expected_comm,
    parse_widths,
    round_energy,
)
from fedavg_app.task import (
    get_device,
    get_model,
    load_partition,
    num_params as width_params,
    add_scaler,
    set_seed,
    static_bn,
    submodel,
    submodel_state,
    test,
)
from fedavg_app.task import train as train_fn

app = ClientApp()

DEADLINE_EXCEEDED = 100  # código de erro enviado ao servidor na política "drop"
BATTERY_DEPLETED = 101  # bateria acabou, ou não sobra o bastante para treinar
ENERGY_SKIPPED = 102  # a política de energia decidiu pular a rodada para economizar


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
    net = get_model(str(rc["model"]), image_size=int(rc["image-size"]),
                    dataset=str(rc["dataset"]))
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainloader, valloader = load_partition(
        partition_id=int(nc["partition-id"]),
        num_partitions=int(nc["num-partitions"]),
        data_dir=str(rc["data-dir"]),
        batch_size=int(rc["batch-size"]),
        method=str(rc["partitioning"]),
        alpha=float(rc["dirichlet-alpha"]),
        seed=int(rc["seed"]),
        dataset=str(rc["dataset"]),
        writers_per_client=int(rc["writers-per-client"]),
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
    num_examples = len(trainloader.dataset)
    server_round = int(msg.content["config"]["server-round"])

    # Energia da rodada: custo fixo + download antes do treino, upload no fim.
    cost, energy_log, budget = None, {}, math.inf
    if use_energy:
        num_params = sum(t.numel() for t in net.parameters())
        cost = round_energy(rc, nc, num_params, server_round)
        energy_log = {
            "energy_baseline": str(rc["energy-baseline"]),
            "energy_policy": str(rc["energy-policy"]),
            "power_factor": float(nc.get("power-factor", 1.0)),
            "net_mbps": float(nc.get("net-mbps", 25.0)),
            "radio_w": float(nc.get("radio-w", 1.8)),
            "bw_down_mbps": cost.bw_down_mbps,
            "bw_up_mbps": cost.bw_up_mbps,
            "battery_before": battery,
        }

    k = num_examples  # amostras treinadas na rodada
    lean = battery is not None and rc["energy-policy"] != "none"
    if lean:
        # Orçamento da rodada = bateria / participações restantes esperadas. Se o
        # cliente pula uma rodada, o que não gastou aumenta o orçamento das próximas.
        left = max(1.0, float(rc["fraction-train"]) * (int(rc["num-server-rounds"])
                                                       - server_round + 1))
        # Sem margem, o orçamento leva a bateria a zero na última participação, e
        # uma rodada de rede ruim perto do fim mata o cliente. tail-margin guarda
        # a cauda da comunicação até a última participação, que a reserva inteira.
        tail = float(rc["tail-margin"])
        last = left <= 1
        margin = expected_comm(nc, num_params, tail) + cost.fixed if tail > 0 else 0.0
        round_budget = battery if last else max(0.0, battery - margin) / left
        # Abaixo de min-data-fraction, a rede domina o custo: guarda a energia para
        # participar depois com mais dados. Na última participação não há depois.
        min_k = float(rc["min-data-fraction"]) * num_examples if not last else 0.0
        epochs = int(rc["local-epochs"])
        compute = cost.per_sample * num_examples * epochs
        estimate = tail if tail > 0 and last else rc["comm-estimate"]
        comm = expected_comm(nc, num_params, estimate)
        eta = data_fraction(str(rc["energy-policy"]), round_budget, compute, comm,
                            cost.fixed)
        if rc["energy-policy"] == "width":
            # HeteroFL: todos os dados, na maior largura que cabe no orçamento.
            arch = (str(rc["model"]), int(rc["image-size"]), str(rc["dataset"]))

            def full_round(w: float) -> float:
                n = width_params(*arch, w)
                c = round_energy(rc, nc, n, server_round, w)
                return (c.per_sample * num_examples * epochs + expected_comm(nc, n, estimate)
                        + c.fixed)

            width = choose_width(parse_widths(rc["widths"]), round_budget, full_round)
            eta = 1.0 if width else 0.0
            energy_log["width"] = width or 0.0
            if width and width < 1.0:
                # O servidor manda o modelo completo; o cliente fica com o recorte
                # e paga a rede só por ele, como se o servidor mandasse só o recorte.
                full = net.state_dict()
                net = submodel(*arch, width)
                net.load_state_dict(submodel_state(full, net))
                num_params = width_params(*arch, width)
                cost = round_energy(rc, nc, num_params, server_round, width)
                comm = expected_comm(nc, num_params, estimate)
                if rc["width-scaler"]:
                    add_scaler(net, width)
            if rc["static-bn"]:
                static_bn(net)  # em todas as larguras, como no HeteroFL
        k = math.floor(eta * num_examples)
        energy_log |= {"round_budget_j": round_budget, "comm_reserved_j": comm,
                       "data_fraction": k / num_examples}
        if k == 0 or k < min_k:
            # Decide ao receber o convite da rodada, antes de baixar o modelo: não gasta.
            _log_metrics(msg, context, "train", {"status": "skipped", **energy_log,
                                                 "num_examples": num_examples,
                                                 "battery_after": battery, "energy_j": 0.0})
            return Message(Error(code=ENERGY_SKIPPED,
                                 reason=f"pulou a rodada (orçamento {round_budget:.1f} J)"),
                           reply_to=msg)
    pre = cost.fixed + cost.download if cost else 0.0
    if battery is not None:
        if battery < pre:
            _set_battery(context, 0.0)
            _log_metrics(msg, context, "train", {"status": "depleted", **energy_log,
                                                 "battery_after": 0.0, "energy_j": battery})
            return _no_battery(msg, "bateria acabou ao baixar o modelo")
    if lean and rc["eta-after-download"] and rc["energy-policy"] != "width":
        # Já baixou: refaz η com o download real e só o upload estimado.
        comm = cost.download + expected_comm(nc, num_params, rc["comm-estimate"]) / 2
        eta = data_fraction(str(rc["energy-policy"]), round_budget, compute, comm,
                            cost.fixed)
        k = math.floor(eta * num_examples)
        energy_log |= {"comm_reserved_j": comm, "data_fraction": k / num_examples}
        if k == 0 or k < min_k:
            # O download saiu caro: desiste antes de treinar e não paga o upload.
            _set_battery(context, battery - pre)
            _log_metrics(msg, context, "train", {"status": "skipped", **energy_log,
                                                 "num_examples": num_examples,
                                                 "energy_comm_j": cost.download,
                                                 "energy_fixed_j": cost.fixed,
                                                 "battery_after": battery - pre,
                                                 "energy_j": pre})
            return Message(Error(code=ENERGY_SKIPPED,
                                 reason=f"pulou a rodada após o download ({pre:.1f} J)"),
                           reply_to=msg)
    if k < num_examples:
        # Subconjunto sorteado a cada rodada (as RNGs foram fixadas em _setup).
        idx = randperm(num_examples)[:k].tolist()
        trainloader = DataLoader(Subset(trainloader.dataset, idx),
                                 batch_size=trainloader.batch_size, shuffle=True)
    if battery is not None:
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
        {"status": status, "num_examples": num_examples, **stats, **energy_log},
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
