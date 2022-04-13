import sys
import os
import importlib
import models
import torch
import torch.nn as nn
import numpy as np
import math
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from transformers import BertTokenizer, BertModel

import torchsparse.nn as spnn
from models.basic_blocks import SparseConvEncoder
from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize, sparse_collate_tensors
importlib.reload(models)

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd(), "models"))  # HACK add the lib folder


class MVT(nn.Module):
    def __init__(self, input_feature_dim=0, args=None):
        super().__init__()
        self.args = args
        self.voxel_size_ap = 0.02
        self.num_classes = 18
      
        self.obj_dim = 512
        self.rotate_number = 4
        self.view_number = 4
        self.rotate_box = True
        self.rotate_obj = True
        self.drop_rate = 0.15

        self.voxel_size = np.array([self.voxel_size_ap]*3)
        self.object_encoder = SparseConvEncoder(input_feature_dim)

        self.language_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.language_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.language_encoder.encoder.layer = self.language_encoder.encoder.layer[:3]

        self.refer_encoder = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=768, nhead=8, activation="gelu"), num_layers=4)

        self.language_clf = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Dropout(self.drop_rate), nn.Linear(768, self.num_classes))
        self.object_language_clf = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Dropout(self.drop_rate), nn.Linear(768, 1))

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.obj_dim, 768),
            nn.LayerNorm(768),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(4, 768),
            nn.LayerNorm(768),
        )

        self.logit_loss = nn.CrossEntropyLoss()
        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=524)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def aug_input(self, box_infos):
        box_infos = box_infos.float().cuda()
        bxyz = box_infos[:,:,:3] # B,N,3
        B,N = bxyz.shape[:2]
        view_theta_arr = torch.Tensor([i*2.0*np.pi/self.view_number for i in range(self.view_number)]).cuda()
        
        bsize = box_infos[:,:,3:6].max(dim=-1).values
        boxs=[]
        for theta in view_theta_arr:
            rotate_matrix = torch.Tensor([[math.cos(theta), -math.sin(theta), 0.0],
                                        [math.sin(theta), math.cos(theta),  0.0],
                                        [0.0,           0.0,            1.0]]).cuda()
            rxyz = torch.matmul(bxyz.reshape(B*N, 3),rotate_matrix).reshape(B,N,3)
            boxs.append(torch.cat([rxyz,bsize.unsqueeze(-1)],dim=-1))
        boxs=torch.stack(boxs,dim=1)
        return boxs

    @torch.no_grad()
    def pre_process(self, data_dict):
        pred_obb_batch = []
        pts_batch = []
        obj_points_batch = []
        class_batch = []
        batch_size = len(data_dict['instance_points'])
        max_num_obj = 0
        for i in range(batch_size):
            instance_point = data_dict['instance_points'][i]
            instance_obb = data_dict['instance_obbs'][i]
            instance_class = data_dict['instance_class'][i]
            num_obj = len(instance_point)
            max_num_obj = max(max_num_obj, num_obj)
            pts = []
            pred_obbs = []
            # filter by class
            for j in range(num_obj):
                pred_obbs.append(torch.Tensor(instance_obb[j]))
                point_cloud = instance_point[j]
                pc = point_cloud[:, :3]

                coords, feats = sparse_quantize(
                    pc,
                    point_cloud,
                    quantization_size=self.voxel_size
                )
                pt = SparseTensor(feats, coords)
                pts.append(pt)
                obj_points_batch.append(point_cloud)
            pts_batch += pts
            pred_obbs = torch.stack(pred_obbs)
            pred_obb_batch.append(pred_obbs)
            class_batch.append(torch.Tensor(instance_class))
        return pts_batch, pred_obb_batch, class_batch, max_num_obj


    def forward(self, data_dict=None):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds,
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        pts_batch, pred_obb_batch, class_batch, max_num_obj = self.pre_process(data_dict)
        B = len(pred_obb_batch)
        feats = sparse_collate_tensors(pts_batch).cuda()
        feats = self.object_encoder(feats)

        #### pad back   
        obj_pad_feats = torch.zeros(B,max_num_obj,512).cuda()
        box_pad_feats = torch.zeros(B,max_num_obj,7).cuda()
        beg = 0
        for idx, bbs in enumerate(pred_obb_batch):
            N = len(bbs)
            obj_pad_feats[idx,:N] = feats[beg:beg+N]
            box_pad_feats[idx,:N] = bbs
            beg+=N
        assert beg == feats.shape[0]
        
        N = max_num_obj
        data_dict["pred_obb_batch"] = box_pad_feats
        obj_feats = self.obj_feature_mapping(obj_pad_feats)
        box_infos = self.aug_input(box_pad_feats)
        box_infos = self.box_feature_mapping(box_infos)
        obj_infos = obj_feats[:, None].repeat(1,self.view_number,1,1) + box_infos

        #### (2) language_encoding ####
        lang_inp = [' '.join(tokens) for tokens in data_dict['lang_token']]
        lang_tokens = self.language_tokenizer(lang_inp, return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()

        lang_infos = self.language_encoder(**lang_tokens)[0]
        LANG_LOGITS = self.language_clf(lang_infos[:,0])
        data_dict["lang_scores"] = LANG_LOGITS

        cat_infos = obj_infos.reshape(B*self.view_number,-1,768)
        mem_infos = lang_infos[:,None].repeat(1,self.view_number,1,1).reshape(B*self.view_number,-1,768)
        out_feats = self.refer_encoder(cat_infos.transpose(0,1),mem_infos.transpose(0,1)).transpose(0,1).reshape(B,self.view_number,-1,768)

        refer_feat = out_feats
        view_feats = (refer_feat / self.view_number).sum(dim=1)
        LOGITS = self.object_language_clf(view_feats).squeeze(-1)
        data_dict["scores"] = LOGITS
        
        return data_dict


