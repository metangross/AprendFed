"""ServerApp: orquestra o FedAvg e avalia o modelo global no conjunto de teste."""

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedavg_app.task import (
    get_device,
    get_model,
    load_centralized_test,
    set_seed,
    test,
)

app = ServerApp()

STRAGGLER_POLICIES = ("wait", "partial", "drop")


def _by_partition(replies: Iterable[Message]) -> list[Message]:
    key = lambda m: int(m.content["info"]["partition-id"]) if m.has_content() else -1
    return sorted(replies, key=key)


class OrderedFedAvg(FedAvg):
    """FedAvg que agrega as respostas na ordem das partições, não na de chegada.

    Somas em ponto flutuante dependem da ordem; sem isso, duas runs com a mesma
    seed divergem levemente a partir da 2ª rodada.
    """

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        return super().aggregate_train(server_round, _by_partition(replies))

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        return super().aggregate_evaluate(server_round, _by_partition(replies))


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    data_dir = str(rc["data-dir"])
    if rc["straggler-policy"] not in STRAGGLER_POLICIES:
        raise ValueError(f"straggler-policy deve ser um de {STRAGGLER_POLICIES}")
    # Data/hora calculadas só aqui e repassadas aos clientes (mesma pasta para todos).
    started = datetime.now()
    run_date, run_time = started.strftime("%d%m%Y"), started.strftime("%H:%M:%S")
    out_dir = Path(str(rc["output-dir"])) / f"run-{context.run_id}-{run_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pesos iniciais do modelo e sorteio de clientes (fraction-train < 1).
    set_seed(int(rc["seed"]))
    device = get_device()

    def new_model(pretrained: bool = False):
        return get_model(str(rc["model"]), pretrained, int(rc["image-size"]))

    testloader = load_centralized_test(data_dir)

    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        net = new_model()
        net.load_state_dict(arrays.to_torch_state_dict())
        loss, acc = test(net, testloader, device)
        return MetricRecord({"test_loss": loss, "test_acc": acc})

    min_nodes = int(rc["min-nodes"])
    strategy = OrderedFedAvg(
        fraction_train=float(rc["fraction-train"]),
        fraction_evaluate=float(rc["fraction-evaluate"]),
        min_train_nodes=min_nodes,
        min_evaluate_nodes=min_nodes,
        min_available_nodes=min_nodes,
    )

    result = strategy.start(
        grid=grid,
        # Com pretrained=true, só o servidor baixa os pesos do ImageNet; os
        # clientes os recebem como modelo global inicial na 1ª rodada.
        initial_arrays=ArrayRecord(new_model(bool(rc["pretrained"])).state_dict()),
        num_rounds=int(rc["num-server-rounds"]),
        train_config=ConfigRecord(
            {"lr": float(rc["lr"]), "run-date": run_date, "run-time": run_time}
        ),
        evaluate_config=ConfigRecord({"run-date": run_date, "run-time": run_time}),
        evaluate_fn=central_evaluate,
    )

    torch.save(result.arrays.to_torch_state_dict(), out_dir / "final_model.pt")
    history = {
        "run_id": context.run_id,
        "run_date": run_date,
        "run_time": run_time,
        "run_config": dict(rc),
        "train_clients": {r: dict(m) for r, m in result.train_metrics_clientapp.items()},
        "evaluate_clients": {
            r: dict(m) for r, m in result.evaluate_metrics_clientapp.items()
        },
        "evaluate_server": {
            r: dict(m) for r, m in result.evaluate_metrics_serverapp.items()
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2))
    print(f"Modelo final e métricas salvos em {out_dir}")
