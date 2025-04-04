import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(run_info):
    train_losses = np.array(run_info["train_losses"]).flatten()
    plt.plot(train_losses, label="Training Loss", color="orange")

def plot_validation_loss(run_info, config):
    val_losses = np.array(run_info["val_losses"]).flatten()
    val_interval = config["VAL_INTERVAL"]
    val_epochs = np.arange(len(val_losses)) * val_interval + val_interval
    plt.plot(val_epochs, val_losses, label="Validation Loss", color="blue")

def plot_losses(run_info, config):
    plot_training_loss(run_info)
    plot_validation_loss(run_info, config)
    plt.title("Training and Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"runs/{config['RUN_ID']}/loss_plot.png")
    plt.show()

def plot_trajectory(t, q, q_pred, config):
    save_path = f"runs/{config['RUN_ID']}/trajectory_plot.png"
    # plot with fading colors as time progresses
    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
    print(t[0], t[-1])
    plt.scatter(q[:, 0], q[:, 1], label="True Trajectory", color="red", marker=".")
    plt.scatter(q_pred[:, 0], q_pred[:, 1], c=colors, label="Predicted Trajectory", marker=".")
    plt.colorbar(label="Time")
    plt.title("True vs Predicted Trajectory")
    plt.xlabel("q1")
    plt.ylabel("q2")
    plt.legend()
    plt.savefig(save_path)
    plt.show()