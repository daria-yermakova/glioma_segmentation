import os
import sys
import argparse
import wandb
import yaml
from yaml.loader import SafeLoader
from datetime import datetime
import numpy as np
import torch

# -- for cluster --
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir[:-21])
print('Cluster root directory', root_dir)
sys.path.append(root_dir)
# -- for cluster --

from src.dataloaders.affine_trans import BRATS
from src.models.unet_without_dropout import UNet
from src.utils import IoU, get_device

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='brats', help="Available [isic, brats]. Default is isic.")

def train(model, epochs, opt, train_loader, val_loader, training_run_name, device):
    path = f'res-{training_run_name}'
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'img'))
        os.makedirs(os.path.join(path, 'target'))
        os.makedirs(os.path.join(path, 'pred'))

    # Log model to wandb
    if W and (not is_sweep):
        wandb.watch(model, log_freq=100)

    print('--amount of epochs--', epochs)
    for epoch in range(epochs):
        train_loss = 0.0
        train_iou = 0.0
        print(f'----------Epoch {epoch}/{epochs}\n', end='')
        for index, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            opt.zero_grad()

            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.0115))
            loss = criterion(logits, targets)
            # loss = loss_function(logits, targets)

            loss.backward()
            opt.step()

            train_loss += float(loss)
            train_iou += float(IoU(targets.detach(), torch.sigmoid(logits.detach()).ge(0.5)))

            if W:
                wandb.log({"training_loss": train_loss / (len(train_loader)),
                           "training_iou": train_iou / (len(train_loader))}, step=epoch)
            if index == 2 and (not is_sweep):
                for batch_index in range(0, len(inputs)):
                    np_img = inputs.cpu().numpy()
                    np_target = targets.cpu().numpy()
                    np_pred = logits.cpu().detach().numpy()
                    np.save(f"{path}/img/img_e{epoch}_b{batch_index}.npy", np_img)
                    np.save(f"{path}/target/target_e{epoch}_b{batch_index}.npy", np_target)
                    np.save(f"{path}/pred/pred_e{epoch}_b{batch_index}.npy", np_pred)

        val_loss = 0
        val_iou = 0
        model.eval()
        with torch.no_grad():
            for index, (inputs, targets, _) in enumerate(val_loader):
                val_inputs, val_targets = inputs.to(device), targets.to(device)
                val_logits = model(val_inputs)

                # loss = loss_function(val_logits, val_targets)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.0115))
                loss = criterion(val_logits, val_targets)

                val_loss += loss.detach().item()
                # val_loss += float(loss)
                val_iou += IoU(val_targets.detach(), torch.sigmoid(val_logits.detach()).ge(0.5))
        if W:
            wandb.log({"validation_loss": val_loss / len(val_loader),
                      "validation_iou": val_iou / len(val_loader)}, step=epoch)
    return model

if __name__ == '__main__':

    args = parser.parse_args()
    dataset = args.dataset

    with open('src/configs/brats.yaml') as f:
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

    device = get_device()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == 'cuda':
        torch.cuda.manual_seed(230)

    # Initialize  U-Net
    unet = UNet().to(device)

    opt = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    # Load Data
    train_set = BRATS('../../../../special-course/data/BRATS_20_images', mode="train", subset=0.6, size=[img_size, img_size, 155])
    valid_set = BRATS('../../../../special-course/data/BRATS_20_images', mode='val', subset=0.6, size=[img_size, img_size, 155])

    # train_set = BRATS('src/BRATS_full_slices', mode="train", subset=0.6, size=[img_size, img_size, 155])
    # valid_set = BRATS('src/BRATS_full_slices', mode='val', subset=0.6, size=[img_size, img_size, 155])


    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # Empty GPU Cache
    torch.cuda.empty_cache()

    # Start Training
    trained_model = train(unet, epochs, opt, train_dataloader, valid_dataloader, training_run_name, device)
    torch.save(trained_model.state_dict(), f'res-{training_run_name}/trained_model.pth')
    if not is_sweep:
        print(
            f"Saved: {training_run_name} Data: {config.dataset}, Model: {config.architecture}")
        # End Training Run
    if W:
        wandb.finish