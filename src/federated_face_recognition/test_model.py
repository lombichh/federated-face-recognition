import torch
from facenet_pytorch import InceptionResnetV1
from federated_face_recognition.task import load_test_data, test

testloader = load_test_data()

model = InceptionResnetV1()
state_dict = torch.load("global_model.pth", map_location="mps" if torch.backends.mps.is_available() else "cpu")
model.load_state_dict(state_dict)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

k = 5
metrics = test(model, testloader, device, k)

print(f"Top-1 Accuracy: {metrics['accuracy_top1']:.4f} ({metrics['accuracy_top1']*100:.2f}%)")
print(f"Top-{k} Accuracy: {metrics['accuracy_topk']:.4f} ({metrics['accuracy_topk']*100:.2f}%)")