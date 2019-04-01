import json
import torch
from torch.utils.data import DataLoader
from input_pipeline import PairsDataset
from model import BicycleGAN

BATCH_SIZE = 2
DATA = '/home/dan/datasets/edges2shoes/train/'
NUM_EPOCHS = 60
DEVICE = torch.device('cuda:0')
MODEL_SAVE_PREFIX = 'models/run01'
TRAIN_LOGS = 'losses_run01.json'
SAVE_STEP = 30000


def main():

    dataset = PairsDataset(folder=DATA, size=256, is_training=True)
    data_loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )
    num_steps = NUM_EPOCHS * (len(dataset) // BATCH_SIZE)
    model = BicycleGAN(device=DEVICE, num_steps=num_steps)
    model.G.load_state_dict(torch.load('models/run00_just_ae_generator.pth'))
    model.E.load_state_dict(torch.load('models/run00_just_ae_encoder.pth'))

    logs = []
    i = 0  # number of weight updates
    text = 'e: {0}, i: {1}, discriminators_only: {2:.3f}, ' +\
           'l1: {3:.3f}, kl: {4:.4f}, lr: {5:.3f}, ' +\
           'fool_d1: {6:.3f}, fool_d2: {7:.3f}, total: {8:.3f}'

    for e in range(NUM_EPOCHS):
        for A, B in data_loader:

            i += 1
            losses = model.train_step(A, B)

            log = text.format(
                e, i, losses['discriminators_loss'],
                losses['l1_loss'], losses['kl_loss'], losses['lr_loss'],
                losses['fool_d1_loss'], losses['fool_d2_loss'],
                losses['total_loss']
            )
            print(log)
            logs.append(losses)

            if i % SAVE_STEP == 0:
                model.save_model(MODEL_SAVE_PREFIX)
                with open(TRAIN_LOGS, 'w') as f:
                    json.dump(logs, f)


main()
