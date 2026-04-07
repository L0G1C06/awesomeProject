import json
import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
import mlflow
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

DATASET_PATH = "reranker_dataset.jsonl"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_PATH = "models/reranker"
EPOCHS = 3


def load_data():
    samples = []
    with open(DATASET_PATH) as f:
        for line in f:
            item = json.loads(line)
            samples.append(
                InputExample(
                    texts=[item["query"], item["doc"]],
                    label=float(item["label"])
                )
            )
    return samples


class MLflowCallback:
    """Loga loss e salva checkpoint a cada epoch."""

    def __init__(self, model, output_path):
        self.model = model
        self.output_path = output_path
        self.epoch = 0
        self.step = 0

    def __call__(self, score, epoch, steps):
        self.epoch = epoch
        self.step = steps

        # Loga a loss no MLflow a cada step
        mlflow.log_metric("loss", score, step=steps)

        # Salva checkpoint ao final de cada epoch (steps == -1 indica fim da epoch)
        if steps == -1:
            checkpoint_path = f"{self.output_path}/checkpoint-epoch-{epoch}"
            self.model.save(checkpoint_path)
            mlflow.log_artifacts(checkpoint_path, artifact_path=f"checkpoint-epoch-{epoch}")
            print(f"  → Checkpoint salvo: epoch {epoch}")


def main():
    print("Loading data...")
    train_samples = load_data()
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

    model = CrossEncoder(MODEL_NAME, num_labels=1)

    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    print("Criando experimento...")
    
    mlflow.set_tracking_uri("http://localhost:5000") 
    mlflow.set_experiment("rag-rerank-training")
    
    experiment = mlflow.get_experiment_by_name("rag-rerank-training")
    print("Experimento:", experiment)

    mlflow.set_experiment("rag-rerank-training")

    print("Starting training...")
    with mlflow.start_run():
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("warmup_steps", 100)

        callback = MLflowCallback(model, OUTPUT_PATH)

        model.fit(
            train_dataloader=train_dataloader,
            epochs=EPOCHS,
            warmup_steps=100,
            callback=callback,          # <-- callback chamado a cada step
        )

        # Salva modelo final
        model.save(OUTPUT_PATH)
        mlflow.log_artifacts(OUTPUT_PATH, artifact_path="model-final")
        print("Modelo final salvo.")


if __name__ == "__main__":
    main()