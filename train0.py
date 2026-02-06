import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.linalg import hadamard
import math
# from CSQ.network import AlexNetFc, ResNet
# from CSQ.options import get_config
# from CSQ.pre_process import build_trans
# from _data import build_loaders, get_class_num, get_topk
from _utils import  hash_center_type

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


def get_hash_centroids(hash_centers, random_centers, labels):
    """
    get labels' hash centroids.
    :param hash_centers: tensor of hash centers
    :param random_centers: tensor of random centers
    :param labels: labels for which hash centroids are to be computed
    :return: tensor of hash centroids
    """
    hash_centroids = []

    for label in labels:
        one_idx = (label == 1).nonzero().squeeze(1)  # find the position of 1 in label
        if len(one_idx) == 0:
            # in some datasets, some image's labels are all zero, we ignore these images
            # let its hash center be zero
            mean_center = torch.zeros((1, hash_centers.size(1)), device=hash_centers.device)
        elif len(one_idx) == 1:
            mean_center = hash_centers[one_idx]
        else:
            mean_center = torch.mean(hash_centers[one_idx], dim=0)
            mean_center[mean_center < 0] = -1
            mean_center[mean_center > 0] = 1
            mean_center[mean_center == 0] = random_centers[mean_center == 0]
            mean_center = mean_center.view(1, -1)

        hash_centroids.append(mean_center)

    return torch.cat(hash_centroids, dim=0)


def build_model(args, pretrained=True):
    if args.backbone == "resnet50" or args.backbone == "resnet152":
        model = ResNet(args, pretrained)
    elif args.backbone == "alexnet":
        model = AlexNetFc(args, pretrained)
    else:
        raise NotImplementedError(f"not support: {args.backbone}")
    return model.cuda()


def train_val(args, train_loader, query_loader, dbase_loader, logger):
    hash_centers = gen_hash_centers(args.n_classes, args.n_bits).cuda()

    random_centers = torch.randint_like(hash_centers[0], 2).cuda()
    random_centers[random_centers == 0] = -1

    model = build_model(args)

    criterion = nn.BCELoss()

    params_list = [
        {"params": model.feature_layers.parameters(), "lr": args.multi_lr * args.lr},
        {"params": model.hash_layer.parameters()},
    ]
    optimizer = torch.optim.Adam(params_list, lr=args.lr, betas=(0.9, 0.999))
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # if len(args.gpu_ids)>1:
    # model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    # model = torch.nn.DataParallel(model).cuda()

    best_map = 0.0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    for epoch in range(args.n_epochs):
        tic = time.time()
        epoch_loss = train_epoch(args, model, train_loader, criterion, hash_centers, random_centers, optimizer, epoch)
        toc = time.time()
        logger.info(
            f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}][loss:{epoch_loss:.4f}]"
        )

        if (epoch + 1) % args.eval_frequency == 0 or (epoch + 1) == args.n_epochs:
            qB, qL = predict(model, query_loader)
            rB, rL = predict(model, dbase_loader)
            map_k = mean_average_precision(qB, rB, qL, rL, args.topk)
            logger.info(
                f"[Evaluating][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][best-mAP@{args.topk}:{best_map}][mAP@{args.topk}:{map_k}][count:{0 if map_k > best_map else (count + 1)}]"
            )

            if map_k > best_map:
                best_map = map_k
                best_epoch = epoch
                best_checkpoint = deepcopy(model.state_dict())
                count = 0
            else:
                count += 1
                if count == 10:
                    logger.info(
                        f"without improvement, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}"
                    )
                    torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map


def train_epoch(args, model, train_loader, criterion, hash_centers, random_centers, optimizer, epoch):
    adjust_learning_rate(args, optimizer, epoch)

    model.train()

    total_loss = []
    for images, labels, _ in train_loader:
        images, labels = images.cuda(), labels.cuda()

        if args.dataset == "cifar":
            # one-hot -> class
            targets = (labels == 1).nonzero()[:, 1]
            # targets = torch.argmax(labels, dim=1)
            hash_centroids = hash_centers[targets]
        else:
            hash_centroids = get_hash_centroids(hash_centers, random_centers, labels)

        y = model(images)

        # {-1,1} -> {0,1}
        center_loss = criterion(0.5 * (y + 1), 0.5 * (hash_centroids + 1))
        # TODO: there's no SimilarityLoss in paper Eq.(5)
        """similarity_loss = pairwise_loss(output1, output2, label1, label2,
                                        sigmoid_param=10. / args.hash_bit,
                                        #l_threshold=15,  # "l_threshold":15.0,
                                        data_imbalance=data_imbalance)"""
        q_loss = torch.mean((torch.abs(y) - 1.0) ** 2)

        loss = center_loss + args.lambda1 * q_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.data.cpu().numpy())

    epoch_loss = np.mean(total_loss)
    return epoch_loss


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.7 ** (epoch // 10))
    # for param_group in optimizer.param_groups:
    # param_group['lr'] = lr
    optimizer.param_groups[0]["lr"] = args.multi_lr * lr
    optimizer.param_groups[1]["lr"] = lr

    return lr


def prepare_loaders(args, bl_fnc):
    train_trans, test_trans = build_trans(args)
    train_loader, query_loader, dbase_loader = bl_fnc(
        args.data_dir,
        args.dataset,
        train_trans,
        test_trans,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    return train_loader, query_loader, dbase_loader


if __name__ == "__main__":
    init("0")

    # python train.py --data_name nus_wide --hash_bit 16 --gpus 2 --model_type Alexnet --lambda1 0  --lambda2 0.001  --R 5000 --eval_frequency 1 --lr 0.0001
    args = get_config()

    dummy_logger_id = None
    rst = []
    for dataset in ["cifar", "nuswide", "flickr", "coco"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)  # used in gen hash-centers
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = prepare_loaders(args, build_loaders)

        for hash_bit in [16, 32, 48, 64, 128]:
            print(f"processing hash-bit: {hash_bit}")
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=False)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", rotation="500 MB", level="INFO")

            with open(f"{args.save_dir}/config.json", "w+") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch, best_map = train_val(args, train_loader, query_loader, dbase_loader, logger)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})
    for x in rst:
        print(
            f"[dataset:{x['dataset']}][bits:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
        )
