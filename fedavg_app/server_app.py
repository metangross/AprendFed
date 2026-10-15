"""ServerApp: orquestra o FedAvg e avalia o modelo global no conjunto de teste."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedavg_app.task import get_device, get_model, load_centralized_test, test

app = ServerApp()

STRAGGLER_POLICIES = ("wait", "partial", "drop")


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    data_dir = str(rc["data-dir"])
    if rc["straggler-policy"] not in STRAGGLER_POLICIES:
        raise ValueError(f"straggler-policy deve ser um de {STRAGGLER_POLICIES}")
    out_dir = Path(str(rc["output-dir"])) / f"run-{context.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
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
    strategy = FedAvg(
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
        train_config=ConfigRecord({"lr": float(rc["lr"])}),
        evaluate_fn=central_evaluate,
    )

    torch.save(result.arrays.to_torch_state_dict(), out_dir / "final_model.pt")
    history = {
        "run_id": context.run_id,
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
