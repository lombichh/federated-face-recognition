import torch
from facenet_pytorch import InceptionResnetV1

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federated_face_recognition.task import get_weights, load_data, set_weights, test, train

class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.model, parameters) # initialize model with global parameters

        # train model
        train_loss = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        return (
            get_weights(self.model),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        print("Evaluating local model...")

        set_weights(self.model, parameters)
        k = 5
        metrics = test(self.model, self.valloader, self.device, k)

        print(f"Local model top-1 Accuracy: {metrics['accuracy_top1']:.4f} ({metrics['accuracy_top1']*100:.2f}%)")
        print(f"Local model top-{k} Accuracy: {metrics['accuracy_topk']:.4f} ({metrics['accuracy_topk']*100:.2f}%)")

        return 0.0, len(self.valloader.dataset), metrics


def client_fn(context: Context):
    print("Loading model...")
    model = InceptionResnetV1()

    print("Loading data...")
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(model, trainloader, valloader, local_epochs).to_client()

app = ClientApp(client_fn=client_fn)
