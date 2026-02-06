import torch
import torch.nn as nn
from torchvision.models import alexnet
# from test import FusionModel , SpatialAttention
from networks.relative_similarity import RelativeSimilarity
from networks.ca_net import *
from utils.attention_zoom import *
from networks.CF_VIT import cf_deit_small

class RelaHash(nn.Module):
    def __init__(self,
                 nbit, nclass, batchsize,
                 init_method='M',
                 pretrained=True, freeze_weight=False,
                 device='cuda',
                 **kwargs):
        super(RelaHash, self).__init__()

        # self.backbone = FusionModel(hash_bit=nbit)

        # self.backbone = CANet(bit=nbit,nclass=nclass)

        # self.config = get_b16_config()
        self.backbone = cf_deit_small(hash_bit=nbit, num_class=nclass)

        # self.backbone.head = nn.Linear(self.backbone.embed_dim, 1000)
        # checkpoint = torch.load('cf-deit-s-7x7-80.8.pth')
        # self.backbone.load_state_dict(checkpoint['model'], strict=False)

        # self.backbone = VisionTransformer(self.config, img_size=448, hash_bit=nbit, nclass=nclass)

        # nn.init.normal_(se.weight, std=0.01)

        # self.hash_fc = nn.Sequential(
        #     nn.BatchNorm1d(self.backbone.num_ftrs // 2 * 3, affine=True),
        #     nn.Linear(self.backbone.num_ftrs // 2 * 3, self.backbone.feature_size),
        #     nn.BatchNorm1d(self.backbone.feature_size, affine=True),
        #     nn.ELU(inplace=True),
        #     nn.Linear(self.backbone.feature_size, nbit),
        # )
        self.relative_similarity = RelativeSimilarity(nbit, nclass, batchsize, init_method=init_method, device=device)

    def get_hash_params(self):
        return list(self.relative_similarity.parameters())
    
    def get_backbone_params(self):
        return self.backbone.get_features_params()
    
    def get_centroids(self):
        return self.relative_similarity.centroids
        
    def forward(self, x):

        ## CANet config
        # f11, f22, f33, y33, feats = self.backbone(x)
        # f44 = torch.cat((f11, f22, f33), -1)
        # code = self.hash_fc(f44)

        ## CF-VIT config
        hash_group, cls_group = self.backbone(x)
        logits = self.relative_similarity(hash_group[-1]) # hashcode


        # return logits, code
        return logits, hash_group, cls_group



# net = RelaHash(nclass=10 , nbit= 16 , batchsize= 32)
# print(net)