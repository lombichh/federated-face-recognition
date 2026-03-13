import torch
from facenet_pytorch import InceptionResnetV1

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from federated_face_recognition.task import get_weights

class FedAvgSaving(FedAvg):
    def __init__(self, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds

    def aggregate_fit(self, server_round, results, failures):
        # aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # last round -> save model
        if aggregated_parameters is not None and server_round == self.num_rounds:
            save_global_model(aggregated_parameters)

        return aggregated_parameters, aggregated_metrics

def weighted_average(metrics):
    """
    Calculate weighted average of client metrics.
    """
    weighted_metrics = {}
    examples = [num_examples for num_examples, _ in metrics]

    for key in metrics[0][1].keys():
        weighted_metrics[key] = sum(metric[key] * num_examples for num_examples, metric in metrics) / sum(examples)

    print("Aggregated metrics:", weighted_metrics)

    return weighted_metrics

def save_global_model(parameters):
    ndarrays = parameters_to_ndarrays(parameters)
    model = InceptionResnetV1()
    
    state_dict = dict(zip(model.state_dict().keys(),
                            [torch.tensor(nd) for nd in ndarrays]))
    model.load_state_dict(state_dict, strict=True)

    torch.save(model.state_dict(), "global_model.pth")
    print("Modello globale salvato su 'global_model.pth'")

def server_fn(context: Context):
    """
    Create server object.
    """
    # read parameters form config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # initialize model parameters
    model = InceptionResnetV1()
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # define strategy
    strategy = FedAvgSaving( # TODO: lookup for other stategies
        num_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)