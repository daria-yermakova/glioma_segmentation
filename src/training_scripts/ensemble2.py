import argparse
import os
import sys
from datetime import datetime
import wandb

import numpy as np
import torch
import yaml
from yaml import SafeLoader

# -- for cluster --
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir[:-21])
print('Cluster root directory', root_dir)
sys.path.append(root_dir)
# -- for cluster --
from src.dataloaders.brats import BRATS
from src.models.unet import UNet
from src.utils import IoU, get_device

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='brats')

def train_ensemble(num_models: 3, epochs, train_loader, path):
    models = []
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'models'))

    for i in range(num_models):
        model = UNet(f'ensemble-{i}')
        if W and (not is_sweep):
            wandb.watch(model, log_freq=100)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0001)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.0115))
        for epoch in range(epochs):
            train_loss = 0.0
            train_iou = 0.0
            print(f'----------Epoch {epoch}/{epochs}\n', end='')
            for index, (inputs, targets, _) in enumerate(train_loader):
                # inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                opt.zero_grad()
                loss = criterion(logits, targets)
                loss.backward()
                opt.step()

                train_loss += float(loss)
                train_iou += float(IoU(targets.detach(), torch.sigmoid(logits.detach()).ge(0.5)))
                if epoch % 5 == 0:
                    torch.save(model.state_dict(), f"{path}/models/model_{i}_{epoch}.pth")
                if W:
                    wandb.log({f"training_loss_{i}": train_loss / (len(train_loader)),
                           f"training_iou_{i}": train_iou / (len(train_loader))}, step=epoch)

                print(f'MODEL: {i} EPOCH: {epoch} INDEX: {index} LOSS: {train_loss/len(train_loader)} IOU: {train_iou /len(train_loader)}')
        models.append(model)
    return models

def ensemble_predict(models, val_loader, path):
    if not os.path.exists(path):
        os.makedirs(path)
    predictions = []
    val_loss = 0
    val_iou = 0
    with torch.no_grad():
        for index, (inputs, targets, _) in enumerate(val_loader):
            batch_predictions = []
            # if not os.path.exists(path):
            os.makedirs(os.path.join(path, f'val_{index}'))
            np_img = inputs.numpy()
            np.save(f"{path}/val_{index}/img_{index}.npy", np_img)
            np_target = targets.detach().numpy()
            np.save(f"{path}/val_{index}/target_{index}.npy", np_target)
            for index_model in range(0, len(models)):
                models[index_model].eval()
                logits = models[index_model](inputs)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.0093))
                loss = criterion(logits, targets)
                val_loss += loss.detach().item()
                val_iou += IoU(targets.detach(), torch.sigmoid(logits.detach()).ge(0.5))
                if W:
                    wandb.log({"validation_loss": val_loss / len(val_loader),
                               "validation_iou": val_iou / len(val_loader)}, step=index)

                np_pred = torch.sigmoid(logits.detach()).ge(0.5).cpu().detach().numpy()
                np.save(f"{path}/val_{index}/pred_{index}_{index_model}.npy", np_pred)
                batch_predictions.append(logits)
            batch_predictions = torch.stack(batch_predictions).mean(dim=0)
            predictions.append(batch_predictions)
    return torch.cat(predictions)

if __name__ == '__main__':

    args = parser.parse_args()
    dataset = args.dataset

    # with open('src/configs/ensemble.yaml') as f:
    with open('../configs/ensemble.yaml') as f:
        default_config = yaml.load(f, Loader=SafeLoader)

    W = default_config['W']
    if W:
        import wandb

        wandb.login()
        # Create the Training Run in Wandb
        wandb.init(project="UnetSegmentation", group='U-net', job_type="Training", config=default_config)
        training_run_name = wandb.run.name
        config = wandb.config
    else:
        # Use current timestamp as name e.g. 2021_12_11_14_46
        training_run_name = str(datetime.now())[:16].replace(" ", "_").replace("-", "_").replace(":", "_")
    print(f"Modelname: {training_run_name}")

    is_sweep = False
    is_sweep = default_config['sweep']
    learning_rate = default_config['learning_rate']
    batch_size = default_config['batch_size']
    epochs = default_config['epochs']
    img_size = default_config['image_size']
    ensemble_size = default_config['ensemble_size']

    device = get_device()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == 'cuda':
        torch.cuda.manual_seed(230)

    # Load Data
    # train_set = BRATS('src/BRATS_full_slices', mode="train", subset=0.6, size=[img_size, img_size, 155])
    # valid_set = BRATS('src/BRATS_full_slices', mode='val', subset=0.6, size=[img_size, img_size, 155])
    train_set = BRATS('../../../../special-course/data/BRATS_20_images', mode="train", subset=0.6, size=[64, 64, 155])
    valid_set = BRATS('../../../../special-course/data/BRATS_20_images', mode='val', subset=0.6, size=[64, 64, 155])

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # Empty GPU Cache
    torch.cuda.empty_cache()

    # Start Training
    path = f'res-{training_run_name}'
    trained_models = train_ensemble(3, epochs, train_dataloader, path)
    ensemble_predict(trained_models, valid_dataloader, path)

    if not is_sweep:
        print(
            f"Saved: {training_run_name} Data: {config.dataset}, Model: {config.architecture}")
        # End Training Run
    if W:
        wandb.finish