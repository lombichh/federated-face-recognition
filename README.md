# Federated Face Recognition

This bachelor thesis project simulates **federated training** of the InceptionResNetV1 Face Recognition model using the dataset Labeled Faces in the Wild.

## Thesis document
The full experimental thesis (PDF) is available here:
* [Read the Thesis](./thesis/thesis_federated_face_recognition_flower.pdf)

## Project

### Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

### Run with the Simulation Engine

In the `federated-face-recognition` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```