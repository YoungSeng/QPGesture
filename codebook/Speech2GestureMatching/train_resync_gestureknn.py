import numpy as np
import os
import torch
import torch.optim as optim
import torch.autograd as autograd

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ResyncNet, Discriminator

from constant import WEIGHT_GEN, WEIGHT_RECON, LAMBDA_GP, GEN_HOP, NUM_MFCC_FEAT, BURNIN_ITER


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def initializer(m, actFun='relu', bn_scale=1.0, bn_bias=0.0):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain(actFun))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, bn_scale)
        nn.init.constant_(m.bias, bn_bias)
    elif isinstance(m, nn.InstanceNorm1d):
        nn.init.constant_(m.weight, bn_scale)
        nn.init.constant_(m.bias, bn_bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain(actFun))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def compute_gradient_penalty(Disc, real_samples, fake_samples, dev):
    N, C, T = real_samples.shape
    alpha = torch.rand((N, 1, 1)).repeat(1, C, T).to(dev)
    
    interpolates = (real_samples * alpha + fake_samples * (1 - alpha)).requires_grad_(True)
    mixed_scores = Disc(interpolates)

    fake = torch.ones_like(mixed_scores)

    gradients = autograd.grad(
        outputs=mixed_scores,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()

    return gradient_penalty


def loss_batch(model_resync, model_disc, train_knn_xb, train_gt_xb, curr_iter, dev, opt_resync=None, opt_disc=None):
    model_disc.zero_grad()

    pred_disc_real = model_disc(train_gt_xb)

    mfcc_knn = train_knn_xb[:, :NUM_MFCC_FEAT]

    pred_gen = model_resync(train_knn_xb).detach()
    pred_knn_xb = torch.cat((mfcc_knn, pred_gen), 1)

    pred_disc_fake = model_disc(pred_knn_xb)

    pred_disc_real_score = -torch.mean(pred_disc_real)
    pred_disc_fake_score = torch.mean(pred_disc_fake)

    if opt_resync is not None:
        gradient_penalty = compute_gradient_penalty(model_disc, train_gt_xb, pred_knn_xb, dev)
        disc_loss = pred_disc_real_score + pred_disc_fake_score + LAMBDA_GP * gradient_penalty

        disc_loss.backward()
        opt_disc.step()
    else:
        disc_loss = pred_disc_real_score + pred_disc_fake_score  
  
    model_resync.zero_grad()

    if (opt_resync is not None and curr_iter % GEN_HOP == 0) or (opt_resync is None):
        mfcc_knn = train_knn_xb[:, :NUM_MFCC_FEAT]

        pred_gen = model_resync(train_knn_xb)
        pred_knn_xb = torch.cat((mfcc_knn, pred_gen), 1)

        recons_loss = nn.L1Loss()(train_knn_xb[:, NUM_MFCC_FEAT:], pred_gen)

        pred_gen_real = model_disc(pred_knn_xb)
        gen_loss = -torch.mean(pred_gen_real)

        total_loss = WEIGHT_GEN * gen_loss + WEIGHT_RECON * recons_loss

        if opt_resync is not None:
            total_loss.backward()
            opt_resync.step()

    return (pred_disc_real_score + pred_disc_fake_score).item()


def fit(num_iters, 
    model_resync, opt_resync, 
    model_disc, opt_disc,
    train_dl,
    model_path, dev):

    best_loss = float('inf')
    curr_iter = 0

    while curr_iter < num_iters:
        print('curr_iter: ' + str(curr_iter))
        for (train_knn_xb, train_gt_xb) in tqdm(train_dl):
            state_info = {
                'iterations': curr_iter,
                'model_resync_state_dict': model_resync.state_dict(),
                'opt_resync_state_dict': opt_resync.state_dict(),
            }

            if curr_iter == num_iters:
                save_model(state_info, os.path.join(model_path, 'latest_model.pth'))
                break
            
            model_resync.train()
            model_disc.train()

            train_knn_xb = train_knn_xb.to(dev, dtype=torch.float)
            train_gt_xb = train_gt_xb.to(dev, dtype=torch.float)

            gen = loss_batch(model_resync, model_disc, train_knn_xb, train_gt_xb, curr_iter, dev, opt_resync, opt_disc)

            if (curr_iter) % 1000 == 0:
                save_model(state_info,  os.path.join(model_path, 'latest_model.pth'))

                curr_criterion = np.abs(gen)

                if curr_criterion < best_loss and curr_iter > BURNIN_ITER:
                    best_loss = curr_criterion
                    save_model(state_info, os.path.join(model_path, 'best_model.pth'))

            curr_iter += 1        


def save_model(state_info, file_name):
    torch.save(state_info, file_name)


def load_resync_model(model_path, dev):
    checkpoint = torch.load(model_path)
    
    model = ResyncNet().cuda(dev)
    opt = optim.Adam(model.parameters(), weight_decay=4e-5)

    model.load_state_dict(checkpoint['model_resync_state_dict'])
    opt.load_state_dict(checkpoint['opt_resync_state_dict'])

    return model, opt


def get_model(lr, dev):
    model_resync = ResyncNet().cuda(dev)
    model_resync.apply(lambda x: initializer(x))

    model_disc = Discriminator().cuda(dev)
    model_disc.apply(lambda x: initializer(x))
    
    return model_resync, optim.Adam(model_resync.parameters(), lr=lr, weight_decay=4e-5, betas=(0.0, 0.9)), \
        model_disc, optim.Adam(model_disc.parameters(), lr=lr, weight_decay=4e-5, betas=(0.0, 0.9))


def train_resync_model(train_ds, bs, num_iters, lr, model_path, dev):
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)

    model_resync, opt_resync, model_disc, opt_disc = \
        get_model(lr, dev)

    fit(num_iters, 
        model_resync, opt_resync,
        model_disc, opt_disc,
        train_dl,
        model_path, dev)