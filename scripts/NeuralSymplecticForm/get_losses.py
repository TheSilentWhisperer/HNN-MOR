import matplotlib.pyplot as plt
import numpy as np

def plot_losses(run_info, config, max_num_points=100):

    train_losses = np.array(run_info["train_losses"]).flatten()
    val_losses = np.concatenate(run_info["val_losses"])
    print(len(val_losses), len(train_losses))

    val_interval = config["VAL_INTERVAL"]
    iterations = np.arange(val_interval-1, len(train_losses), val_interval)

    train_losses = train_losses[iterations]
    step_size = len(train_losses) // max_num_points

    step_size = max(1, step_size)

    iterations = iterations[::step_size]
    train_losses = train_losses[::step_size]
    val_losses = val_losses[::step_size]

    plt.plot(iterations, train_losses, label="Train Loss", color="blue")
    plt.plot(iterations, val_losses, label="Validation Loss", color="orange")

    plt.yscale("log")

    plt.title("Training and Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"runs/{config['RUN_ID']}/loss_plot.png")
    plt.show()

if __name__ == "__main__":
    
    import os
    import json
    
    config = json.load(open("scripts/NeuralSymplecticForm/config.json"))
    run_id = config["RUN_ID"]
    run_info_path = os.path.join("runs", run_id, "run_info.json")
    config_path = os.path.join("runs", run_id, "config.json")
    run_info = json.load(open(run_info_path))
    config = json.load(open(config_path))
    plot_losses(run_info, config)
    print("Plot saved at runs/{}/loss_plot.png".format(run_id))
