import os

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from src import MIR1K, E2E0, cycle, summary, SAMPLE_RATE, bce
from evaluate import evaluate


def train():
    logdir = 'runs/Hybrid_bce'

    hop_length = 160

    learning_rate = 5e-4
    batch_size = 16
    validation_interval = 2000
    clip_grad_norm = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MIR1K('Hybrid', hop_length, ['train'], whole_audio=False, use_aug=True)
    validation_dataset = MIR1K('Hybrid', hop_length, ['test'], whole_audio=True, use_aug=False)

    data_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=2)
    
    iterations = 200000
    learning_rate_decay_steps = 2000
    learning_rate_decay_rate = 0.98
    resume_iteration = None

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    model = E2E0(4, 1, (2, 2)).to(device)
    if resume_iteration is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model_{resume_iteration}.pt')
        ckpt = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(ckpt['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    summary(model)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    RPA, RCA, OA, VFA, VR, it = 0, 0, 0, 0, 0, 0

    for i, data in zip(loop, cycle(data_loader)):
        mel = data['mel'].to(device)
        pitch_label = data['pitch'].to(device)
        pitch_pred = model(mel)
        loss = bce(pitch_pred, pitch_label)

        print(i, end='\t')
        print('loss_total:', loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        scheduler.step()
        writer.add_scalar('loss/loss_pitch', loss.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                metrics = evaluate(validation_dataset, model, hop_length, device)
                for key, value in metrics.items():
                    writer.add_scalar('stage_pitch/' + key, np.mean(value), global_step=i)
                rpa = np.mean(metrics['RPA'])
                rca = np.mean(metrics['RCA'])
                oa = np.mean(metrics['OA'])
                vr = np.mean(metrics['VR'])
                vfa = np.mean(metrics['VFA'])
                RPA, RCA, OA, VR, VFA, it = rpa, rca, oa, vr, vfa, i
                with open(os.path.join(logdir, 'result.txt'), 'a') as f:
                    f.write(str(i) + '\t')
                    f.write(str(RPA) + '\t')
                    f.write(str(RCA) + '\t')
                    f.write(str(OA) + '\t')
                    f.write(str(VR) + '\t')
                    f.write(str(VFA) + '\n')
                torch.save({'model': model.state_dict()}, os.path.join(logdir, f'model_{i}.pt'))
            model.train()

if __name__ == '__main__':
    train()
