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
        # 1. 前端网络特征提取：COP 产生 [37, 77]，AOP 产生 [17, 17]
        if is_rgb_available:
            self.current_feature_map = self.cop_net.step(rgb_frame)  # [B, 512, 37, 77]
            
        sd_feat = self.sd_net.step(sd_slice)  # 原生 160x160 输入 -> 提取出 [B, 512, 17, 17]
        td_feat = self.td_net.step(td_slice)  # 原生 160x160 输入 -> 提取出 [B, 512, 17, 17]
        
        # 2. 混合单元接口（HUs）融合
        dvs_raw_fuse = (sd_feat + td_feat) / 2.0
        feature_delta = self.hu_fuse(dvs_raw_fuse)  # 保持低通道开销的 [B, 512, 17, 17]
        
        if self.current_feature_map is None:
            self.current_feature_map = torch.zeros((feature_delta.shape[0], 512, 37, 77), device=feature_delta.device)
            
        # 3. 🚀 关键物理对齐：在得到特征后，再强行重采样对齐空间尺寸
        # 完美复现 1个 AOP 像素格在全局画布上对应一个 2x4 密集 COP 像素块的物理本质！
        feature_delta_aligned = F.interpolate(
            feature_delta, 
            size=(self.current_feature_map.shape[2], self.current_feature_map.shape[3]), 
            mode='nearest'
        )  # 规整至 [B, 512, 37, 77]
        
        self.current_feature_map = self.current_feature_map + feature_delta_aligned
        
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
