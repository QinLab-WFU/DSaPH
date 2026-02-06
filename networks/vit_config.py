import ml_collections


def get_b16_config():
    """
    about model
    :return:
    """
    c = ml_collections.ConfigDict()
    c.patches = ml_collections.ConfigDict({'size': (16, 16)})
    c.split = 'non-overlap'
    c.slide_step = 12
    c.hash_length =32
    c.hidden_size = 768
    c.transformer = ml_collections.ConfigDict()
    c.transformer.mlp_dim = 3072
    c.transformer.num_heads = 12
    c.transformer.num_layers = 12

    c.transformer.attention_dropout_rate = 0.0
    c.transformer.dropout_rate = 0.1
    c.classifier = 'token'
    c.representation_size = None

    c.radius = 2.0
    c.gamma = 10.0
    c.beta = 5.0
    c.hypseed = 0
    c.topk = None
    c.output_dim = 32
    c.device= 'cuda:0'
    c.epochs = 60
    c.dataset = "UCMDmutil"
    c.eta = 0.2
    c.n_workers=4
    c.batch_size=2
    c.p = 4
    # c.data_dir ="../ _datasets"
    return c


config = get_b16_config()
vit_pretrain = 'ViT-B_16.npz'  # pretrained weights for backbone, please edit it for your configuration
epochs = 60
lr = 3e-4

zoom_size = 512
input_size = 448
batch_size = 2
dataset = 'UCMDmutil'
if dataset == 'dfc15':
    train_csv = '/home/g/桌面/YG/HASH-ZOO/_datasets/dfc15/train.txt'
    test_csv = '/home/g/桌面/YG/HASH-ZOO/_datasets/dfc15/train.txt'
    train_root = '/home/g/桌面/YG/HASH-ZOO/_datasets/dfc15/train.txt'
    test_root = '/home/g/桌面/YG/HASH-ZOO/_datasets/dfc15/train.txt'
    config.num_classes = 8
elif dataset == 'mlrs':
    train_csv = '/home/g/桌面/YG/HASH-ZOO/_datasets/mlrs/train.txt'
    test_csv = '/home/g/桌面/YG/HASH-ZOO/_datasets/mlrs/query.txt'
    train_root = '/home/g/桌面/YG/HASH-ZOO/_datasets/mlrs/image'
    test_root = '/home/g/桌面/YG/HASH-ZOO/_datasets/mlrs/image'
    config.num_classes = 60
elif dataset == 'UCMDmutil':
    train_csv = '/home/g/桌面/YG/HASH-ZOO/_datasets/UCMDmutil/train.txt'
    test_csv = '/home/g/桌面/YG/HASH-ZOO/_datasets/UCMDmutil/query.txt'
    train_root = '/home/g/桌面/YG/HASH-ZOO/_datasets/UCMDmutil/images'
    test_root = '/home/g/桌面/YG/HASH-ZOO/_datasets/UCMDmutil/images'
    config.num_classes = 17

else:
    assert False, 'no dataset'



lr_ml = 1
alpha = 1

beta = [1, 1, 1, 1]



CUDA_VISIBLE_DEVICES = '0,1,2,3'
momentum = 0.9
weight_decay = 5e-4
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

weight_path = 'checkpoints/UCMD-20.pth'  # weight for val.py
