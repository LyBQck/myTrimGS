import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import os
import torch.utils.data as Data
from sdf_network import SdfNet

def training(data, model, loss_fn, optimizer, num_epoch=50, print_step=5, save_dir='output'):

    pbar = tqdm(total=num_epoch)
    pbar.set_description('Training')
    for epoch in range(1, num_epoch + 1):

        train_loss = 0
        for imgs, labels in data:
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if epoch == 1 or epoch % print_step == 0 or epoch == num_epoch:
            with open (save_dir + '/loss.txt', 'a') as f:
                f.write(f'Epoch: {epoch}, Loss: {train_loss / len(data)}\n')
        
        if epoch % 5 == 0:
            pbar.update(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='nn_sdf/data/scan24')
    parser.add_argument('--out_dir', type=str, default='nn_sdf/output')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--print_step', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    stl = np.load(f'{args.dataset_dir}/stl.npy')
    neg = np.load(f'{args.dataset_dir}/neg.npy')
    dist = np.load(f'{args.dataset_dir}/neg_dist.npy')
    x = np.concatenate([stl, neg], axis=0)
    y = np.concatenate([np.zeros((len(stl), 1)), dist], axis=0)

    # test_num = 100
    # args.batch_size = 10
    # x = np.arange(test_num).reshape(test_num, 1)
    # x = np.concatenate([x, x, x], axis=1)
    # y = np.arange(test_num).reshape(test_num, 1)
    dataset = Data.TensorDataset(torch.FloatTensor(x).to(device), torch.FloatTensor(y).to(device))
    train_set, test_set = Data.random_split(dataset,
                                                lengths=[int(0.7 * len(dataset)),
                                                len(dataset) - int(0.7 * len(dataset))],
                                                generator=torch.Generator().manual_seed(0))
    # trainLoader = DataLoader(dataset = train_set.dataset, batch_size = args.batch_size, shuffle = True)
    trainLoader = DataLoader(dataset = train_set, batch_size = args.batch_size, shuffle = True)
    testLoader = DataLoader(dataset = test_set, batch_size = args.batch_size, shuffle = True)

    net = SdfNet(latent_dim=args.latent_dim).to(device)
    # loss_fn = nn.MSELoss()
    # nn.L1Loss()
    loss_fn = nn.SmoothL1Loss()
    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    training(trainLoader, net, loss_fn, optimizer, num_epoch=args.num_epoch, print_step=args.print_step, save_dir=args.out_dir)
    torch.save(net.state_dict(), args.out_dir + '/model.pth')
