import torch
import os


def save_model(model, epoch, optimizer, multiple_gpus, save_path):
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    # Please refer this link for saving and loading models
    # Link: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save({
        'epoch' : epoch,
        # Please change this..... if you are using single GPU to model.state_dict().
        'model_state_dict': model.module.state_dict() if multiple_gpus == True else model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict()
    }, f'{save_path}{epoch}.pth')
    print(f'Weight saved for epoch {epoch}.')