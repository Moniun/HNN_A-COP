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

    def get_oracle_rgb_feature(self, current_rgb_frame):
        """
        🚀 为自监督训练特供的教师特征接口
        直接利用当前的真实 RGB 提取纯语义无状态特征图，作为优化特征融合路径的 Ground Truth 特征
        """
        with torch.no_grad(): # 特征对其预训练时，教师分支不计算梯度，保持稳定
            oracle_feat = self.cop_net.step(current_rgb_frame)
        return oracle_feat

    def forward(self, rgb_frame, sd_slice, td_slice, is_rgb_available=True):
        if is_rgb_available:
            self.current_feature_map = self.cop_net.step(rgb_frame)  # [B, 512, 37, 77]
            
        sd_feat = self.sd_net.step(sd_slice)  # [B, 512, 17, 17]
        td_feat = self.td_net.step(td_slice)  # [B, 512, 17, 17]
        
        # 混合单元空间与通道对齐融合
        dvs_raw_fuse = (sd_feat + td_feat) / 2.0
        feature_delta = self.hu_fuse(dvs_raw_fuse)  # [B, 512, 17, 17]
        
        if self.current_feature_map is None:
            self.current_feature_map = torch.zeros((feature_delta.shape[0], 512, 37, 77), device=feature_delta.device)
            
        # 空间维度最近邻插值拉伸靠齐 (完美表达 1个AOP格 对应 2x4个COP格)
        feature_delta_aligned = F.interpolate(
            feature_delta, 
            size=(self.current_feature_map.shape[2], self.current_feature_map.shape[3]), 
            mode='nearest'
        )  # 规整至 [B, 512, 37, 77]
        
        self.current_feature_map = self.current_feature_map + feature_delta_aligned
        
        return self.current_feature_map


class TaskHead(nn.Module):
    """保持原样，为后续的任务二、任务三留好标准的无缝衔接接口"""
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
        self.head = TaskHead(in_channels=512, num_objects=num_objects)
    
    def reset_stream_state(self):
        self.backbone.reset_stream_state()
    
    def forward(self, rgb_frame, sd_slice, td_slice, is_rgb_available=True):
        feat = self.backbone(rgb_frame, sd_slice, td_slice, is_rgb_available)
        pred = self.head(feat)
        return pred