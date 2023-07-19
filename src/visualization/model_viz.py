import pandas as pd
import matplotlib.pyplot as plt


def visualize_training_metrics(
    logs_save_path: str, num_folds: int, version_num: int = 0
) -> None:
    for fold in range(0, num_folds):
        metrics = pd.read_csv(
            f"{logs_save_path}/logs_f{fold}/lightning_logs/version_{version_num}/metrics.csv"
        )
        del metrics["step"]
        metrics = metrics.fillna(method="ffill")
        metrics = metrics.drop(metrics.index[0::2])
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].plot(
            metrics["epoch"], metrics["val_loss"], "-o", label="Validation Loss"
        )
        axs[0].plot(
            metrics["epoch"], metrics["Train_loss"], "-o", label="Training Loss"
        )
        axs[1].plot(
            metrics["epoch"], metrics["val_dice"], "-o", label="Validation Dice"
        )

        axs[0].legend()
        axs[1].legend()

        axs[0].set_title(f"Metrics for Fold {fold+1}")
        axs[1].set_title(f"Metrics for Fold {fold+1}")

        plt.show()
