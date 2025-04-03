import torch
from tqdm import tqdm

def train_for_one_epoch(model, train_loader, val_loader, lr):
    """
    Train the model for one epoch and validate it.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_loss, val_loss = 0, 0

    model.train()
    for q, dq, ddq in tqdm(train_loader):
        u = torch.cat((q, dq), dim=-1)
        du = torch.cat((dq, ddq), dim=-1)
        du_pred = model(u)
        loss = criterion(du_pred, du)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), lr)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for q, dq, ddq in tqdm(val_loader):
            u = torch.cat((q, dq), dim=-1)
            du = torch.cat((dq, ddq), dim=-1)
            du_pred = model(u)
            loss = criterion(du_pred, du)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    return train_loss, val_loss

if __name__ == "__main__":

    from torch.utils.data import DataLoader, random_split
    from utils.data.DynDataset import DynDataset
    from models.NeuralSymplecticForm import NeuralSymplecticForm

    config = {
        "DATA_PATH": "datasets/2d_pend/unact_small_dq0.pickle",
        "BATCH_SIZE": 64,
        "LR": 0.01,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = DynDataset(config["DATA_PATH"], device = device, data_type="point")
    train_dataset, val_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.1), int(len(dataset) * 0.1)])
    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True), DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False), DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)

    model = NeuralSymplecticForm(dataset.manifold_dim).to(device)

    for _ in range(10):
        train_loss, val_loss = train_for_one_epoch(model, train_loader, val_loader, config["LR"])
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")