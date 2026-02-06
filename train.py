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
    lg2 = 0 if n_bits < 1 else int(math.log(n_bits, 2))
    if 2**lg2 != n_bits:
        return "random"

    if n_classes <= n_bits:
        return "ha_d"
    elif n_classes > n_bits and n_classes <= 2 * n_bits:
        return "ha_2d"
    else:
        return "random"


def gen_hash_centers(n_classes, n_bits):
    t = hash_center_type(n_classes, n_bits)
    if t == "ha_d":
        ha_d = torch.from_numpy(hadamard(n_bits)).float()
        hash_centers = ha_d[0:n_classes]
    elif t == "ha_2d":
        ha_d = torch.from_numpy(hadamard(n_bits)).float()
        hash_centers = torch.cat((ha_d, -ha_d), 0)[0:n_classes]
    elif t == "random":
        prob = torch.ones(n_classes, n_bits) * 0.5
        hash_centers = torch.bernoulli(prob) * 2.0 - 1.0
    else:
        raise NotImplementedError
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

        for i in input_size_list:
            resized_img = F.interpolate(data,
                                        (i, i),
                                        mode='bilinear',
                                        align_corners=True
                                        )
            resized_img = torch.squeeze(resized_img)

            if (resized_img.ndim == 3):
                resized_img = resized_img.unsqueeze(0)

            sample_list.append(resized_img)
        sample_list.append(data)
        # logits, code = model(sample_list)





        # loss = LOSS(code, labels.float())



        logits, hash_g, cls_g = model(sample_list)

        code_1, code_2 = hash_g[0], hash_g[1]
        cls_1, cls_2 = cls_g[0], cls_g[1]

        # print("cls_1:", cls_1.shape)
        # print("cls_2:", cls_2.shape)

        loss1 = LOSS(code_1, labels.float())
        loss2 = LOSS(code_2, labels.float())

        loss3 = LOSS2(cls_1, labels.float())
        loss4 = LOSS2(cls_2, labels.float())
        # print("hash0:", hash_g[0])
        # print("hash1:", hash_g[1])

        loss0 = criterion(code_2, lab_hash, labels.float())

        # loss = 1.5 * loss1 + 1.5 * loss2 + 0.5 * loss3 +0.5* loss4
        loss = 0.5 * loss1 + 0.5 * loss2

        # loss =  0.5 * loss3 + 0.5 * loss4

        loss.backward()
        optimizer.step()
        # loss = loss2




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
        # ``````````````````````
        # data = data.to(device)
        # labels = labels.to(device)
        # ``````````````````````
        with torch.no_grad():
            data, labels = data.to(device), labels.to(device) # hen trainning MLRS delete this line

            ## CF_VIT process
            sample_list = []

            for i in input_size_list:
                resized_img = F.interpolate(data,
                                            (i, i),
                                            mode='bilinear',
                                            align_corners=True
                                            )
                resized_img = torch.squeeze(resized_img)

                if (resized_img.ndim == 3):
                    resized_img = resized_img.unsqueeze(0)

                sample_list.append(resized_img)

            sample_list.append(data)

            logits, hash_g, cls_g = model(sample_list)

            code_1, code_2 = hash_g[0], hash_g[1]
            cls_1, cls_2 = cls_g[0], cls_g[1]
            loss1 = LOSS(code_1, labels.float())
            loss2 = LOSS(code_2, labels.float())

            loss3 = LOSS2(cls_1, labels.float())
            loss4 = LOSS2(cls_2, labels.float())


            loss0 = criterion(code_2, lab_hash, labels.float())
            loss = 0.5 * loss1 + 0.5 * loss2 + 0*loss0
            # loss = 1.5* loss1 + 1.5* loss2 + 0.5* loss3 + 0.5 * loss4

            # loss = 0.5 * loss3 + 0.5 * loss4

            hamm_dist = get_hamm_dist(code_2, centroids, normalize=True) # 哈希码

            acc, cbacc = calculate_accuracy(logits, hamm_dist, labels, loss_param['multiclass'])

            if return_codes:
                ret_codes.append(code_2)
                ret_labels.append(labels.float())

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



## MLRS 数据transform
def prepare_dataloader(config):
    logging.info('Creating Datasets')
    if config['dataset'] == 'MLRS' :
        MLRSs.init('./data/MLRS/', 1000, 5000)
        transform = transforms.Compose([
            transforms.Resize(224), # 都改成224
            transforms.RandomCrop(224), # 改成224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = MLRSs('./data/', 'train',transform=transform)
        testset = MLRSs('./data/', 'query',transform=transform)
        database = MLRSs('./data/', 'retrieval',transform=transform)
        train_loader = DataLoader(trainset, config['batch_size'])
        test_loader =DataLoader(testset, config['batch_size'], shuffle=False, drop_last=False)
        db_loader = DataLoader(database, config['batch_size'], shuffle=False, drop_last=False)
        return train_loader ,test_loader , db_loader
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

    ## HYP2
    # sheet = xlrd.open_workbook('codetable.xlsx').sheet_by_index(0)
    sheet = xlrd.open_workbook('codetable.xls').sheet_by_index(0)
    threshold = sheet.row(nbit)[math.ceil(math.log(nclass, 2))].value

    Loss1 = HyP(num_classes=nclass, num_bits=nbit, device=device, threshold=threshold)
    Loss2 = nn.CrossEntropyLoss()
    # Loss3 =Triplet(model.hash_g, model.cls_g,label)
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


        # `````````````````````````````````````````````````````````````````````````````````
        # cam_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).cuda()
        cam_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).cuda()
        grad_cam = GradCAM2(cam_model, cam_model.layer4[-1])




        # cam_model = GradCAM2(model, [model.backbone.blocks[-1],model.backbone.norm,model.backbone.head])

        # train_meters = train_hashing(optimizer, model, centroids, train_loader, loss_param, LOSS=Loss1,LOSS2=Loss2,cam_model=cam_model)
        train_meters = train_hashing(config,optimizer,optimizer_add, model, centroids, train_loader, loss_param, LOSS=Loss1, LOSS2=Loss2,grad_cam=grad_cam)
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
                # pth占内存！！！！！！！！！
                io.fast_save(modelsd, f'{logdir}/models/best.pth')
                io.fast_save(optimsd, f'{logdir}/optims/best.pth')
                # pth占内存！！！！！！！！！
                if config['wandb_enable']:
                    wandb.run.summary["best_map"] = best



        json.dump(train_history, open(f'{logdir}/train_history.json', 'w+'), indent=True, sort_keys=True)
        # io.fast_save(train_outputs, f'{logdir}/outputs/train_last.pth')

        if len(test_history) != 0:
            json.dump(test_history, open(f'{logdir}/test_history.json', 'w+'), indent=True, sort_keys=True)
            # io.fast_save(test_outputs, f'{logdir}/outputs/test_last.pth')

        save_now = config['save_interval'] != 0 and (ep + 1) % config['save_interval'] == 0
        # pth占内存！！！！！！！！！
        if save_now:
            io.fast_save(modelsd, f'{logdir}/models/ep{ep + 1}.pth')
            io.fast_save(optimsd, f'{logdir}/optims/ep{ep + 1}.pth')
            # pth占内存！！！！！！！！！
            # io.fast_save(train_outputs, f'{logdir}/outputs/train_ep{ep + 1}.pth')

        if best < curr_metric:
            best = curr_metric
            # pth占内存！！！！！！！！！
            io.fast_save(modelsd, f'{logdir}/models/best.pth')
            #
    modelsd = model.state_dict()
    # pth占内存！！！！！！！！！
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
