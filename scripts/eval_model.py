import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import sys
sys.path.append('./')
from metrics import find_metrics, find_metrics_macro
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == 'cpu':
    print('Models will run on cpu.')

class CreateDataset(Dataset):
    def __init__(self, csv_file, encoder, image_size, return_label = True):
        self.csv_file = csv_file
        self.images = csv_file['updated_paths'].values.tolist()
        self.labels = csv_file['species'].values.tolist()
        self.encoder = encoder
        self.encoded_labels = self.encoder.transform(np.array(self.labels).reshape(-1, 1)).toarray()
        self.convert_rgb = torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        self.return_label = return_label
        self.image_size = image_size
    
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img = torchvision.io.read_file(self.images[idx])
        img = torchvision.io.decode_jpeg(img)
        if img.shape[0] != 3:
            img = self.convert_rgb(img)
        img = torchvision.transforms.functional.resize(img, (self.image_size[0], self.image_size[1]))
        img = img / 255.0
        if self.return_label:
            return img, torch.Tensor(self.encoded_labels[idx])
        else:
            return img


class EvalModel():
    def __init__(self, csv_file, model_weights, encoder, image_size):
        self.cross_entropy = nn.CrossEntropyLoss()
        self.model = self.init_model(model_weights)
        self.dataset = self.init_dataloaders(csv_file, encoder, image_size)

    def init_model(self, model_weights):
        incep_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained = False)
        incep_model.fc = nn.Linear(2048, 30)
        if model_weights != None:
            incep_model.load_state_dict(torch.load(model_weights)['model_state_dict'])
        incep_model.eval()
        return incep_model
    
    def init_dataloaders(self, dataset, encoder, image_size):
        dataset = CreateDataset(dataset, encoder, image_size)
        return DataLoader(dataset, batch_size = 2, shuffle = False, num_workers = 4, prefetch_factor=2)

    def itereate_dataset(self, dataset, model):
        loss_, prec_, rec_ = 0.0, 0.0, 0.0
        for img, labels in tqdm(dataset):
            img = img.to(device)
            labels = labels.to(device)
            outputs = model(img)
            loss = self.cross_entropy(outputs, labels)
            prec, rec = find_metrics_macro(outputs, labels, batch_size=64)
            loss_ += loss.item()
            prec_ += prec.item()
            rec_ += rec.item()
        return loss_, prec_, rec_
        
    def eval_mdoel(self):
        self.model = self.model.to(device)
        loss, prec, rec = self.itereate_dataset(self.dataset, self.model)
        print(f'Loss: {loss / len(self.dataset)}')
        print(f'Prec: {prec / len(self.dataset)}')
        print(f'Rec: {rec / len(self.dataset)}')