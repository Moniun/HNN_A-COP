import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ann_model import ResNet2Stage
from models.snn_model import ResNet2StageSNN


class TianmoucHNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cop_net = ResNet2Stage(inchannel=3, block_num=[1, 1])
        self.sd_net = ResNet2StageSNN(firstchannels=64, channels=(64, 128), inchannel=2, block_num=[1, 1])
        self.td_net = ResNet2StageSNN(firstchannels=64, channels=(64, 128), inchannel=1, block_num=[1, 1])
        
        self.hu_fuse = nn.Sequential(
            nn.Conv2d(128 * 4, 128 * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(128 * 4),
            nn.ReLU(inplace=True)
        )
        
        self.register_buffer("current_feature_map", None, persistent=False)

    def reset_stream_state(self):
        self.current_feature_map = None
        self.sd_net.reset_state(history=False)
        self.td_net.reset_state(history=False)

    def forward(self, rgb_frame, sd_slice, td_slice, is_rgb_available=True):
        if is_rgb_available:
            self.current_feature_map = self.cop_net.step(rgb_frame)
            
        sd_feat = self.sd_net.step(sd_slice)
        td_feat = self.td_net.step(td_slice)
        
        # 先使用一种简单高效的融合方式，后续进行优化
        dvs_raw_fuse = (sd_feat + td_feat) / 2.0
        feature_delta = self.hu_fuse(dvs_raw_fuse)
        
        if self.current_feature_map is None:
            self.current_feature_map = torch.zeros_like(feature_delta)
            
        self.current_feature_map = self.current_feature_map + feature_delta
        
        return self.current_feature_map


class DetectionHead(nn.Module):
    def __init__(self, in_channels=512, num_objects=3):
        super().__init__()
        self.num_objects = num_objects
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_objects * 2)
        )

    def forward(self, feature_map):
        b = feature_map.shape[0]
        x = self.global_pool(feature_map).view(b, -1)
        out = self.fc(x).view(b, self.num_objects, 2)
        return out


class FullModel(nn.Module):
    def __init__(self, num_objects=3):
        super().__init__()
        self.backbone = TianmoucHNNBackbone()
        self.head = DetectionHead(in_channels=512, num_objects=num_objects)
    
    def reset_stream_state(self):
        self.backbone.reset_stream_state()
    
    def forward(self, rgb_frame, sd_slice, td_slice, is_rgb_available=True):
        feat = self.backbone(rgb_frame, sd_slice, td_slice, is_rgb_available)
        pred = self.head(feat)
        return pred
