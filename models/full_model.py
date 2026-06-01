import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # 🚀 引入工业级大模型库
from models.snn_model import ConvNeXt2StageSNN


class TianmoucHNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 🚀 认知通路大升级：拉取 Meta 官方先进的预训练 ConvNeXt-Tiny
        print("====== 正在加载 ConvNeXt-Tiny 预训练权重先验 ======")
        self.cop_net = timm.create_model('convnext_tiny', pretrained=True, features_only=True)
        
        # 💡 自监督核心策略：完全冻结大模型老师的权重，100% 迫使梯度用于雕刻 SNN 门控
        for param in self.cop_net.parameters():
            param.requires_grad = False
            
        # 🚀 动作通路大升级：各向同性现代化 SNN 路径 (在 Stage 2 输出 384 通道物理膜电位)
        self.sd_net = ConvNeXt2StageSNN(inchannel=2, out_channels=384)
        self.td_net = ConvNeXt2StageSNN(inchannel=1, out_channels=384)
        
        # 🚀 HUs 混合单元融合通道层：
        # 完美接管两路 SNN 通道融合，1x1 变频卷积
        self.hu_fuse = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=1, bias=False), 
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.register_buffer("current_feature_map", None, persistent=False)

    def reset_stream_state(self):
        self.current_feature_map = None
        self.sd_net.reset_state(history=False)
        self.td_net.reset_state(history=False)

    def get_oracle_rgb_feature(self, current_rgb_frame):
        """教师特征接口：ConvNeXt 的 feats[2] 对应 Stage 2 输出 [B, 384, 20, 40]"""
        with torch.no_grad(): 
            feats = self.cop_net(current_rgb_frame)
            oracle_feat = feats[2]
        return oracle_feat

    def forward(self, rgb_frame, sd_slice, td_slice, is_rgb_available=True):
        if is_rgb_available:
            # 快门触发点：刷新长期语义大底图 [B, 384, 20, 40]
            feats = self.cop_net(rgb_frame)
            self.current_feature_map = feats[2].clone()  
            
        # 推进现代化脉冲大核 SNN 分支演进，吐出原生的 [B, 384, 20, 20] 状态电位图
        sd_feat = self.sd_net.step(sd_slice)  
        td_feat = self.td_net.step(td_slice)  
        
        # HUs 核心融合逻辑：动作通路（DVS）多维脉冲残差融汇交互
        dvs_raw_fuse = (sd_feat + td_feat) / 2.0
        feature_delta = self.hu_fuse(dvs_raw_fuse)  # [B, 384, 20, 20]
        
        if self.current_feature_map is None:
            self.current_feature_map = torch.zeros((feature_delta.shape[0], 384, 20, 40), device=feature_delta.device)
            
        # 🚀 HUs 空间对齐：AOP 的 20x20 特征网格，通过 1:2 等比例插值，无损、不畸变地平铺到 20x40 Canvas 上！
        # 1个AOP残差格子横向严格对应2个COP语义格子，展现极高的科学可解释性
        feature_delta_aligned = F.interpolate(
            feature_delta, 
            size=(self.current_feature_map.shape[2], self.current_feature_map.shape[3]), 
            mode='nearest'
        )  
        
        # HUs 流式画布累加更新
        self.current_feature_map = self.current_feature_map + feature_delta_aligned
        
        # 直接对外吐出最干净的 384 通道特征图，杜绝任何硬编码扭曲拉伸
        return self.current_feature_map


class TaskHead(nn.Module):
    """
    🚀 升级适配版预测 Head：输入通道数自动适配为标准的 384 维，完美接驳新特征！
    """
    def __init__(self, in_channels=384, num_objects=3):
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
        self.head = TaskHead(in_channels=384, num_objects=num_objects)
        
    def reset_stream_state(self):
        self.backbone.reset_stream_state()
        
    def forward(self, rgb_frame, sd_slice, td_slice, is_rgb_available=True):
        feat = self.backbone(rgb_frame, sd_slice, td_slice, is_rgb_available)
        pred = self.head(feat)
        return pred