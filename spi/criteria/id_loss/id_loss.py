import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self, path_ir_se50, num_scales=1):
        super(IDLoss, self).__init__()
        # print(f'Loading ResNet ArcFace from: {path_ir_se50}')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(path_ir_se50))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.num_scales = num_scales

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def calculate_similarity(self, x, y):
        assert x.shape[0] == 1
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)
        simmilar = x_feats[0].dot(y_feats[0])
        return simmilar

    def calculate_batch_similarity(self, x, y):
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # torch.Size([100, 512]) torch.Size([100, 512])
        simmilar = x_feats * y_feats
        simmilar = torch.sum(simmilar, dim=-1)
        simmilar = torch.mean(simmilar)
        return simmilar

    def forward(self, x, y):
        n_samples = x.shape[0]
        loss = 0.0
        for _scale in range(self.num_scales):
            x_feats = self.extract_feats(x)
            y_feats = self.extract_feats(y)
            for i in range(n_samples):
                diff_target = y_feats[i].dot(x_feats[i])
                loss += 1 - diff_target
            
            if _scale != self.num_scales - 1:
                x = F.interpolate(x, mode='bilinear', scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
                y = F.interpolate(y, mode='bilinear', scale_factor=0.5, align_corners=False, recompute_scale_factor=True)
        return loss / n_samples

    def psp_forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs
