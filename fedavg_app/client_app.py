"""ClientApp: treinamento e avaliação locais na GPU."""

import json
import time
from pathlib import Path

from flwr.app import (
    ArrayRecord,
    Context,
    Error,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from fedavg_app.task import get_device, get_model, load_partition, test
from fedavg_app.task import train as train_fn

app = ClientApp()

DEADLINE_EXCEEDED = 100  # código de erro enviado ao servidor na política "drop"


def _setup(msg: Message, context: Context):
    """Instancia o modelo com os pesos globais e carrega a partição local."""
    rc, nc = context.run_config, context.node_config
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


def _log_metrics(msg: Message, context: Context, phase: str, metrics: dict) -> None:
    """Acrescenta uma linha em outputs/run-<id>/clients/partition-<p>.jsonl."""
    pid = int(context.node_config["partition-id"])
    record = {
        "run_id": context.run_id,
        "round": int(msg.content["config"]["server-round"]),
        "phase": phase,
        "partition_id": pid,
        "node_id": context.node_id,
        "compute_speed": float(context.node_config.get("compute-speed", 1.0)),
        "timestamp": time.time(),
        **metrics,
    }
    out = Path(str(context.run_config["output-dir"])) / f"run-{context.run_id}" / "clients"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"partition-{pid}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


@app.train()
def train(msg: Message, context: Context) -> Message:
    device = get_device()
    net, trainloader, _ = _setup(msg, context)
    rc = context.run_config
    policy = str(rc["straggler-policy"])
    deadline = float(rc["round-deadline"]) if policy != "wait" else 0.0

    stats = train_fn(
        net,
        trainloader,
        epochs=int(rc["local-epochs"]),
        lr=float(msg.content["config"]["lr"]),
        momentum=float(rc["momentum"]),
        device=device,
        compute_speed=float(context.node_config.get("compute-speed", 1.0)),
        deadline=deadline,
    )
    completed = stats.pop("completed")
    status = "ok" if completed else ("dropped" if policy == "drop" else "partial")
    _log_metrics(
        msg,
        context,
        "train",
        {"status": status, "num_examples": len(trainloader.dataset), **stats},
    )

    if status == "dropped":
        # Não cumpriu o prazo: o servidor conta como falha e ignora este cliente.
        reason = f"prazo de {deadline}s excedido (sim_time={stats['sim_time_s']:.2f}s)"
        return Message(Error(code=DEADLINE_EXCEEDED, reason=reason), reply_to=msg)

    # Peso no FedAvg = exemplos efetivamente processados, então um cliente que
    # treinou parcialmente ("partial") contribui proporcionalmente menos.
    metrics = MetricRecord(
        {"train_loss": stats["train_loss"], "num-examples": stats["examples_processed"]}
    )
    content = RecordDict({"arrays": ArrayRecord(net.state_dict()), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
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
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
