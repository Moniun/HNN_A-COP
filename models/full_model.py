import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  
from models.snn_model import ConvNeXt2StageSNN


class TianmoucHNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        print("====== 正在加载 ConvNeXt-Tiny 预训练权重先验 ======")
        self.cop_net = timm.create_model('convnext_tiny', pretrained=True, features_only=True)
        for param in self.cop_net.parameters():
            param.requires_grad = False
            
        self.dvs_net = ConvNeXt2StageSNN(inchannel=3, out_channels=384)
        
        self.hu_fuse_dvs = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=1, bias=False), 
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

    def reset_stream_state(self):
        self.dvs_net.reset_state(history=False)

    def get_oracle_rgb_feature(self, current_rgb_frame):
        with torch.no_grad(): 
            feats = self.cop_net(current_rgb_frame)
            oracle_feat = feats[2]
        return oracle_feat

    def forward(self, rgb_frame, sd_slice, td_slice, is_rgb_available=True):
        if is_rgb_available:
            feats = self.cop_net(rgb_frame)
            ann_feat = feats[2].clone()
        else:
            ann_feat = torch.zeros((sd_slice.shape[0], 384, 20, 40), device=sd_slice.device)
            
        dvs_input = torch.cat([sd_slice, td_slice], dim=1)
        dvs_feat = self.dvs_net.step(dvs_input)  
        
        dvs_feat_aligned = F.interpolate(dvs_feat, size=(20, 40), mode='nearest')
        
        feature_delta_dvs = self.hu_fuse_dvs(dvs_feat_aligned)
        
        output_feat = ann_feat + feature_delta_dvs
        return output_feat


# 修改 models/full_model.py 内部的 TaskHead
class TaskHead(nn.Module):
    """
    🚀 学术级全卷积解耦检测头 (Decoupled Task Head)
    保持 20x40 空间特征分辨率，通过独立的卷积分支解耦位置定位，彻底抹除非前景物体的误触发！
    """
    def __init__(self, in_channels=384, num_objects=3):
        super().__init__()
        self.num_objects = num_objects
        
        # 共享特征前置层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 专攻边界框回归的分支通路
        self.reg_convs = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 每个空间网格位置密集吐出 3 * 4 个预测坐标通道
        self.reg_pred = nn.Conv2d(256, num_objects * 4, kernel_size=1)
        
        # 自适应空间池化，用于将 20x40 的精细空间响应映射到指定的 Few-Shot 物体数量上
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, feature_map):
        # 输入 feature_map Shape: [B, 384, 20, 40]
        x = self.stem(feature_map)  # -> [B, 256, 20, 40]
        
        reg_feat = self.reg_convs(x)
        reg_out = self.reg_pred(reg_feat)  # -> [B, num_objects * 4, 20, 40]
        
        # 通过自适应池化聚合空间局部特征
        reg_out = self.spatial_pool(reg_out).view(feature_map.shape[0], self.num_objects, 4)
        # 吐出标准的 [B, num_objects, 4] 相对坐标
        return reg_out


class FullModel(nn.Module):
    def __init__(self, num_objects=3):
        super().__init__()
        self.backbone = TianmoucHNNBackbone()
        self.head = TaskHead(in_channels=384, num_objects=num_objects)
        
    def reset_stream_state(self):
        self.backbone.reset_stream_state()
        
    def forward(self, rgb_frame, sd_slice, td_slice, is_rgb_available=True):
        feat = self.backbone(rgb_frame, sd_slice, td_slice, is_rgb_available)
        pred = self.head(feat)
        return pred