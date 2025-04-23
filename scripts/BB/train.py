import torch
from tqdm import tqdm
import sys

def train_for_one_epoch(epoch, optimizer, model, train_loader, val_loader, config):
    """
    Train the model for one epoch and validate it.
    """
    criterion = torch.nn.MSELoss()

    train_loss = []
    val_loss = []

    model.train()
    with tqdm(train_loader, file=sys.stdout) as pbar:
        for q, dq, ddq, tau in train_loader:
            ddq_pred = model(q, dq, tau)
            loss = criterion(ddq_pred, ddq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            iter = epoch * len(train_loader) + len(train_loss)

            if iter % config["VAL_INTERVAL"] == 0:
                sum_loss = 0
                model.eval()
                for q, dq, ddq, tau in val_loader:
                    ddq_pred = model(q, dq, tau)
                    loss = criterion(ddq_pred, ddq)
                    sum_loss += loss.item()
                val_loss.append(sum_loss / len(val_loader))
                model.train()

                if val_loss[-1] < config["BEST_LOSS"]:
                    config["BEST_LOSS"] = val_loss[-1]
                    with open(os.path.join("runs", config["RUN_ID"], "config.json"), "w") as f:
                        json.dump(config, f, indent=4)
                    torch.save(model.state_dict(), os.path.join("runs", config["RUN_ID"], f"model.pt"))

            pbar.set_description(f"Epoch {epoch} | Train Loss: {train_loss[-1]:.4f} | Val Loss: {val_loss[-1] if val_loss else 0:.4f}")
            pbar.update()
        

    return train_loss, val_loss

def train(model, train_loader, val_loader, config):
    run_info = {
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "train_losses": [],
        "val_losses": []
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    for epoch in range(config["NUM_EPOCHS"]):
        train_loss, val_loss = train_for_one_epoch(epoch, optimizer, model, train_loader, val_loader, config)
        run_info["train_losses"].append(train_loss)
        run_info["val_losses"].append(val_loss)
    return run_info

if __name__ == "__main__":

    torch.manual_seed(0)

    from torch.utils.data import DataLoader
    from utils.data.DynDataset import DynDataset
    from models.BB import BB
    import json
    from datetime import datetime
    import os

    config = json.load(open("scripts/BB/config.json"))
    data_config = json.load(open("scripts/data_config.json"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = DynDataset(data_config["DATA_PATH"], device, num_traj = data_config["NUM_TRAJ"], data_type="point", split="train", keys=["q", "dq", "ddq", "tau"])
    val_dataset = DynDataset(data_config["DATA_PATH"], device, num_traj = data_config["NUM_TRAJ"], data_type="point", split="val", keys=["q", "dq", "ddq", "tau"])

    train_loader, val_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True), DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)

    model = BB(train_dataset.manifold_dim).to(device)

    #create a directory to save the new run
    if "RUN_ID" in config and os.path.exists(os.path.join("runs", config["RUN_ID"])):
        config = json.load(open(f"runs/{config['RUN_ID']}/config.json"))
        model.load_state_dict(torch.load(os.path.join("runs", config["RUN_ID"], "model.pt")))
    else:
        config["BEST_LOSS"] = float("inf")
    config["RUN_ID"] = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(os.path.join("runs", config["RUN_ID"]))
    # overwrite the config file
    with open(f"scripts/BB/config.json", "w") as f:
        json.dump(config, f, indent=4)

    #save the config file
    with open(os.path.join("runs", config["RUN_ID"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    run_info = train(model, train_loader, val_loader, config)
    # overwrite the config file
    with open(f"scripts/BB/config.json", "w") as f:
        json.dump(config, f, indent=4)
    # save the run info
    with open(os.path.join("runs", config["RUN_ID"], "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=4)