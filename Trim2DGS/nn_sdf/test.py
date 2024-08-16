import torch
import numpy as np
from torch import nn
import argparse
from torch.utils.data import DataLoader
import torch.utils.data as Data
from sdf_network import SdfNet

def validate(data, model, loss_fn):
    print("Validating:\n")
    with torch.no_grad():
        val_loss = 0
        for imgs, labels in data:
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            val_loss += loss
        
        print("Val loss:{}\n".format(val_loss / len(data)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='nn_sdf/data/scan24')
    parser.add_argument('--out_dir', type=str, default='nn_sdf/output')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=10240)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)

    stl = np.load(f'{args.dataset_dir}/stl.npy')
    neg = np.load(f'{args.dataset_dir}/neg.npy')
    dist = np.load(f'{args.dataset_dir}/neg_dist.npy')
    x = np.concatenate([stl, neg], axis=0)
    y = np.concatenate([np.zeros((len(stl), 1)), dist], axis=0)

    dataset = Data.TensorDataset(torch.FloatTensor(x).to(device), torch.FloatTensor(y).to(device))
    train_set, test_set = Data.random_split(dataset,
                                                lengths=[int(0.7 * len(dataset)),
                                                len(dataset) - int(0.7 * len(dataset))],
                                                generator=torch.Generator().manual_seed(0))
    trainLoader = DataLoader(dataset = train_set, batch_size = args.batch_size, shuffle = True)
    testLoader = DataLoader(dataset = test_set, batch_size = args.batch_size, shuffle = True)


    sdf_net = SdfNet(latent_dim=args.latent_dim).to("cuda")
    load_dir = args.out_dir + '/model.pth'
    sdf_net.load_state_dict(torch.load(load_dir))
    sdf_net.eval()

    validate(testLoader, sdf_net, nn.SmoothL1Loss())
# [i for i in sdf_net.parameters()]