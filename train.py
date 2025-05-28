import math
from venv import logger

import torchvision.models
import xlrd
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pprint import pprint

from huggingface_hub import resume_inference_endpoint
from sympy.abc import theta
# from scipy.special import label
from torchvision import transforms
import torch

from torch.optim import Adam
import math
from torch.utils.data import DataLoader
from transformers.image_transforms import resize

import configs
from hashing.utils import calculate_accuracy, get_hamm_dist, calculate_mAP
from networks.loss import RelaHashLoss
from networks.model import RelaHash
from utils import io
from utils1 import load_pretrained

from utils.misc import AverageMeter, Timer
from tqdm import tqdm
from networks.triplet_loss import TripletLoss
from utils.attention_zoom import *
from utils.datasets import MLRSs
from loss.hyp import HyP
# from networks.gradcam import GradCAM2
from networks.network import GradCAM2
from torchvision import models
from loss0 import DFPHLoss
from scipy.linalg import hadamard
from train0 import gen_hash_centers
from _utils import build_optimizer
# ```````````````````````````````````````````````````````````````````````````````````````````````````````````````
# def train_hashing(optimizer, model, centroids, train_loader, loss_param, LOSS=None, LOSS2=None,cam_model=None):
def try_hadamard(config):
    centers = gen_hash_centers(n_classes=17, n_bits=config['nbit'])
    return centers.cuda()

def hash_center_type(n_classes, n_bits):
    """
    used in CenterHashing, CSQ, ...
    """
    


def gen_hash_centers(n_classes, n_bits):
    '''
 
     '''
    print(f"hash center type: {t}, shape: {hash_centers.shape}")
    return hash_centers


def train_hashing(config,optimizer,optimizer_add, model, centroids, train_loader, loss_param, LOSS=None, LOSS2=None, LOSS3= None, grad_cam=None):
    model.train()
    device = loss_param['device']
    nclass = loss_param['arch_kwargs']['nclass']
    nbit = loss_param['arch_kwargs']['nbit']
    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()
    criterion=DFPHLoss(config)
    lab_hash = try_hadamard(config)

    total_timer.tick()
    input_size_list = [112, 224]
    pbar = tqdm(train_loader, desc='Training', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (data, labels) in enumerate(pbar):
        timer.tick()
        # ``````````````````````
        # data = data.to(device)
        # labels = labels.to(device)
        # ``````````````````````

        # clear gradient
        optimizer.zero_grad()



        data, labels = data.to(device), labels.to(device) # hen trainning MLRS delete this line
        sample_list = []
        # ```````````````````````````````````````````````
        cam_data = grad_cam.calculate_cam(data).detach()
        data = data + 1.2 * cam_data

        '''
     
         '''
        logits, hash_g, cls_g = model(sample_list)

        '''
     
         '''
        
       


        hamm_dist = get_hamm_dist(code_2, centroids, normalize=True)
        acc, cbacc = calculate_accuracy(logits, hamm_dist, labels, loss_param['multiclass'])

        timer.toc()
        total_timer.toc()

        # store results
        meters['loss_total'].update(loss.item(), data.size(0))
        meters['acc'].update(acc.item(), data.size(0))
        meters['cbacc'].update(cbacc.item(), data.size(0))

        meters['time'].update(timer.total)

        pbar.set_postfix({'Train_loss': meters['loss_total'].avg,
                            'A(CE)': meters['acc'].avg,
                            'A(CB)': meters['cbacc'].avg})

    print()
    total_timer.toc()

    meters['total_time'].update(total_timer.total)

    return meters


def test_hashing(config,model, centroids, test_loader, loss_param, return_codes=False, LOSS=None, LOSS2=None,LOSS3=None):
    model.eval()
    device = loss_param['device']
    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    criterion = DFPHLoss(config)
    timer = Timer()
    lab_hash = try_hadamard(config)
    nclass = loss_param['arch_kwargs']['nclass']
    nbit = loss_param['arch_kwargs']['nbit']
    total_timer.tick()
    input_size_list = [112, 224]
    ret_codes = []
    ret_labels = []

    pbar = tqdm(test_loader, desc='Test', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (data, labels) in enumerate(pbar):
        timer.tick()
     
        with torch.no_grad():
            data, labels = data.to(device), labels.to(device) # hen trainning MLRS delete this line

          
            sample_list = []

            for i in input_size_list:
                '''
     
                 '''
        timer.toc()
        total_timer.toc()

        # store results
        meters['loss_total'].update(loss.item(), data.size(0))
        meters['acc'].update(acc.item(), data.size(0))
        meters['cbacc'].update(cbacc.item(), data.size(0))

        meters['time'].update(timer.total)

        pbar.set_postfix({'Eval_loss': meters['loss_total'].avg,
                            'A(CE)': meters['acc'].avg,
                            'A(CB)': meters['cbacc'].avg})

    print()
    meters['total_time'].update(total_timer.total)

    if return_codes:
        res = {
            'codes': torch.cat(ret_codes),
            'labels': torch.cat(ret_labels)
        }
        return meters, res

    return meters


    '''
     dataset preprocessing
     '''

    # num_train, num_test, num_database = len(trainset), len(testset), len(database)
    else :
        train_dataset = configs.dataset(config, filename='train.txt', transform_mode='train')

        separate_multiclass = config['dataset_kwargs'].get('separate_multiclass', False)
        config['dataset_kwargs']['separate_multiclass'] = False
        test_dataset = configs.dataset(config, filename='test.txt', transform_mode='test')
        db_dataset = configs.dataset(config, filename='database.txt', transform_mode='test')
        config['dataset_kwargs']['separate_multiclass'] = separate_multiclass  # during mAP, no need to separate

        logging.info(f'Number of DB data: {len(db_dataset)}')
        logging.info(f'Number of Train data: {len(train_dataset)}')

        print(train_dataset)

        train_loader = configs.dataloader(train_dataset, config['batch_size'])
        test_loader = configs.dataloader(test_dataset, config['batch_size'], shuffle=False, drop_last=False)
        db_loader = configs.dataloader(db_dataset, config['batch_size'], shuffle=False, drop_last=False)
        return train_loader, test_loader, db_loader





def main(config):
    Best_map = 0
    device = torch.device(config.get('device', 'cuda:0'))

    io.init_save_queue()

    start_time = time.time()
    configs.seeding(config['seed'])

    logdir = config['logdir']
    assert logdir != '', 'please input logdir'

    pprint(config)

    if config['wandb_enable']:
        import wandb
        ## initiaze wandb ##
        wandb_dir = logdir
        wandb.init(project="relahash", config=config, dir=wandb_dir)
        # wandb run name
        wandb.run.name = logdir.split('logs/')[1]


    os.makedirs(f'{logdir}/models', exist_ok=True)
    os.makedirs(f'{logdir}/optims', exist_ok=True)
    os.makedirs(f'{logdir}/outputs', exist_ok=True)
    json.dump(config, open(f'{logdir}/config.json', 'w+'), indent=4, sort_keys=True)

    nclass = config['arch_kwargs']['nclass']
    nbit = config['arch_kwargs']['nbit']

    train_loader, test_loader, db_loader = prepare_dataloader(config)
    model = RelaHash(**config['arch_kwargs'])
    model.to(device)
    # hyp = HyP(nbit,nclass)
    # Triplet = TripletLoss()
    print(model)

    logging.info(f'Total Bit: {nbit}')
    centroids = model.get_centroids()
    io.fast_save(centroids, f'{logdir}/outputs/centroids.pth')

    if config['wandb_enable']:
        wandb.watch(model)

       '''
     
         '''
    backbone_lr_scale = 1
    optimizer = Adam([
            {'params': model.get_backbone_params(), 'lr': config['optim_kwargs']['lr'] * backbone_lr_scale},
            {'params': model.get_hash_params()},
            # {'params': Loss1.parameters(),'lr': 5e-4}
            {'params': Loss1.parameters(), 'lr': 0.00005}
        ],
        lr=config['optim_kwargs']['lr'],
        betas=config['optim_kwargs'].get('betas', (0.9, 0.999)),
        weight_decay=config['optim_kwargs'].get('weight_decay', 0))
    scheduler = configs.scheduler(config, optimizer)
    optimizer_add = build_optimizer(config['optim'], model.parameters(), lr=config['optim_kwargs']['lr'], weight_decay=config['optim_kwargs']['weight_decay'])
    train_history = []
    test_history = []

    loss_param = config.copy()
    loss_param.update({'device': device})

    best = 0
    curr_metric = 0
    theta= config['theta']
    eta =config['eta']
    gamma=config['gamma']

    nepochs = config['epochs']
    neval = config['eval_interval']

    logging.info('Training Start')

    load_pretrained(config,model.backbone,logger)
    z_time=0
    for ep in range(nepochs):
        stime=time.time()
        logging.info(f'Epoch [{ep + 1}/{nepochs}]')
        res = {'ep': ep + 1}
       
        '''
     
         '''
        scheduler.step()
        z_time += time.time() - stime
        logging.info(f'z time used: {z_time:.4f} ')

        for key in train_meters: res['train_' + key] = train_meters[key].avg
        train_history.append(res)

        # train_outputs.append(train_out)
        if config['wandb_enable']:
            wandb_train = res.copy()
            wandb_train.pop("ep")
            wandb.log(wandb_train, step=res['ep'])

        modelsd = model.state_dict()
        optimsd = optimizer.state_dict()


        eval_now = (ep + 1) == nepochs or (neval != 0 and (ep + 1) % neval == 0)
        if eval_now:
            res = {'ep': ep + 1}

            db_meters, db_out = test_hashing(config,model, centroids, db_loader, loss_param, True,LOSS=Loss1, LOSS2=Loss2)
            test_meters, test_out = test_hashing(config,model, centroids, test_loader, loss_param, True, LOSS=Loss1, LOSS2=Loss2)

            for key in test_meters: res['test_' + key] = test_meters[key].avg
            for key in db_meters: res['db_' + key] = db_meters[key].avg

            res['mAP'] = calculate_mAP(db_out['codes'], db_out['labels'],
                                       test_out['codes'], test_out['labels'],
                                       loss_param['R'],ep=ep)
            logging.info(f'mAP: {res["mAP"]:.6f}')

            print('mAP : %.6f', res['mAP'])
            if res['mAP'] > Best_map :
                Best_map = res['mAP']
            print(f'Best mAP : {Best_map}')

            curr_metric = res['mAP']
            test_history.append(res)
            # test_outputs.append(outs)

            if config['wandb_enable']:
                wandb_test = res.copy()
                wandb_test.pop("ep")
                wandb.log(wandb_test, step=res['ep'])
            if best < curr_metric:
                best = curr_metric
                io.fast_save(modelsd, f'{logdir}/models/best.pth')
                io.fast_save(optimsd, f'{logdir}/optims/best.pth')
                if config['wandb_enable']:
                    wandb.run.summary["best_map"] = best



        json.dump(train_history, open(f'{logdir}/train_history.json', 'w+'), indent=True, sort_keys=True)
        # io.fast_save(train_outputs, f'{logdir}/outputs/train_last.pth')

        if len(test_history) != 0:
            json.dump(test_history, open(f'{logdir}/test_history.json', 'w+'), indent=True, sort_keys=True)
            # io.fast_save(test_outputs, f'{logdir}/outputs/test_last.pth')

        save_now = config['save_interval'] != 0 and (ep + 1) % config['save_interval'] == 0
        if save_now:
            io.fast_save(modelsd, f'{logdir}/models/ep{ep + 1}.pth')
            io.fast_save(optimsd, f'{logdir}/optims/ep{ep + 1}.pth')
            # pth占内存！！！！！！！！！
            # io.fast_save(train_outputs, f'{logdir}/outputs/train_ep{ep + 1}.pth')

        if best < curr_metric:
            best = curr_metric
            io.fast_save(modelsd, f'{logdir}/models/best.pth')
            #
    modelsd = model.state_dict()
    io.fast_save(modelsd, f'{logdir}/models/last.pth')
    io.fast_save(optimsd, f'{logdir}/optims/last.pth')
    #
    total_time = time.time() - start_time
    io.join_save_queue()
    logging.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')

    logging.info(f'Best mAP: {best:.6f}')
    print(f'Epoch : {ep}  Best mAP: {best:.6f}')
    logging.info(f'Done: {logdir}')

    return logdir
