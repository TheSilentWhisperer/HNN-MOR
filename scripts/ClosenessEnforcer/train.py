import torch
from tqdm import tqdm
import sys

def train_for_one_epoch(epoch, optimizer, model, train_loader, val_loader, config):
    """
    Train the model for one epoch and validate it.
    """
    criterion = torch.nn.MSELoss()

    prediction_train_loss = []
    prediction_val_loss = []
    closeness_train_loss = []
    closeness_val_loss = []

    model.train()
    with tqdm(train_loader, file=sys.stdout) as pbar:
        for q, dq, ddq in train_loader:
            u = torch.cat((q, dq), dim=-1)
            du = torch.cat((dq, ddq), dim=-1)
            du_pred = model(u)
            prediction_loss = criterion(du_pred, du)
            closeness_loss = model.get_closeness_loss(u)

            prediction_train_loss.append(prediction_loss.item())
            closeness_train_loss.append(closeness_loss.item())
            loss = prediction_loss + config["CLOSENESS_LOSS_WEIGHT"] * closeness_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["GRAD_CLIP"])
            optimizer.step()

            iter = epoch * len(train_loader) + len(prediction_train_loss)

            if iter % config["VAL_INTERVAL"] == 0:
                sum_prediction_loss = 0
                sum_closeness_loss = 0
                model.eval()
                with torch.no_grad():
                    for q, dq, ddq in val_loader:
                        u = torch.cat((q, dq), dim=-1)
                        du = torch.cat((dq, ddq), dim=-1)
                        du_pred = model(u)
                        prediction_loss = criterion(du_pred, du)
                        closeness_loss = model.get_closeness_loss(u)
                        sum_prediction_loss += prediction_loss.item()
                        sum_closeness_loss += closeness_loss.item()
                prediction_val_loss.append(sum_prediction_loss / len(val_loader))
                closeness_val_loss.append(sum_closeness_loss / len(val_loader))
                model.train()

                last_val_loss = prediction_val_loss[-1] + config["CLOSENESS_LOSS_WEIGHT"] * closeness_val_loss[-1]

                if last_val_loss < config["BEST_LOSS"]:
                    config["BEST_LOSS"] = last_val_loss
                    with open(os.path.join("runs", config["RUN_ID"], "config.json"), "w") as f:
                        json.dump(config, f, indent=4)
                    torch.save(model.state_dict(), os.path.join("runs", config["RUN_ID"], f"model.pt"))

            pbar.set_description(f"Epoch {epoch} | Train Loss (pred, close): {prediction_train_loss[-1]:.4f}, {closeness_train_loss[-1]:.4f} | Val Loss (pred, close): {prediction_val_loss[-1] if prediction_val_loss else 0:.4f}, {closeness_val_loss[-1] if closeness_val_loss else 0:.4f}")
            pbar.update()
        
    return prediction_train_loss, prediction_val_loss, closeness_train_loss, closeness_val_loss

def train(model, train_loader, val_loader, config):
    run_info = {
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "prediction_train_losses": [],
        "closeness_train_losses": [],
        "prediction_val_losses": [],
        "closeness_val_losses": []
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    for epoch in range(config["NUM_EPOCHS"]):
        prediction_train_loss, prediction_val_loss, closeness_train_loss, closeness_val_loss = train_for_one_epoch(epoch, optimizer, model, train_loader, val_loader, config)
        run_info["prediction_train_losses"].append(prediction_train_loss)
        run_info["closeness_train_losses"].append(closeness_train_loss)
        run_info["prediction_val_losses"].append(prediction_val_loss)
        run_info["closeness_val_losses"].append(closeness_val_loss)

    return run_info

if __name__ == "__main__":

    torch.manual_seed(0)

    from torch.utils.data import DataLoader
    from utils.data.DynDataset import DynDataset
    from models.ClosenessEnforcer import ClosenessEnforcer
    import json
    from datetime import datetime
    import os

    config = json.load(open("scripts/ClosenessEnforcer/config.json"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = DynDataset(config["DATA_PATH"], device, num_traj=config["NUM_TRAJ"], data_type="point", split="train", keys=["q", "dq", "ddq"])
    val_dataset = DynDataset(config["DATA_PATH"], device, num_traj=config["NUM_TRAJ"], data_type="point", split="val", keys=["q", "dq", "ddq"])

    train_loader, val_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True), DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)

    model = ClosenessEnforcer(train_dataset.manifold_dim).to(device)

    #create a directory to save the new run
    if "RUN_ID" in config and os.path.exists(os.path.join("runs", config["RUN_ID"])):
        config = json.load(open(f"runs/{config['RUN_ID']}/config.json"))
        model.load_state_dict(torch.load(os.path.join("runs", config["RUN_ID"], "model.pt")))
    else:
        config["BEST_LOSS"] = float("inf")
    config["RUN_ID"] = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(os.path.join("runs", config["RUN_ID"]))
    # overwrite the config file
    with open(f"scripts/ClosenessEnforcer/config.json", "w") as f:
        json.dump(config, f, indent=4)

    #save the config file
    with open(os.path.join("runs", config["RUN_ID"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    run_info = train(model, train_loader, val_loader, config)
    # overwrite the config file
    with open(f"scripts/ClosenessEnforcer/config.json", "w") as f:
        json.dump(config, f, indent=4)
    # save the run info
    with open(os.path.join("runs", config["RUN_ID"], "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=4)