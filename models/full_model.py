import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  
from models.snn_model import ConvNeXt2StageSNN


class SignalQualityGate(nn.Module):
    """
    🚀 信号质量自适应决策头 (自平滑优化版)：
    引入时域门控记忆单元与一阶惯性滤波，防止权重在时间片跨步时发生断崖式阶跃突变。
    """
    def __init__(self, in_channels=384):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 3, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3, bias=False)
        )
        # 🔒 核心优化：缓存上一步的注意力系数，实现时域平滑
        self.register_buffer("last_weights", None, persistent=False)

    def forward(self, cop_feat, sd_feat, td_feat, training_mode=True):
        b, c, h, w = cop_feat.shape
        w_cop = self.global_pool(cop_feat).view(b, -1)
        w_sd = self.global_pool(sd_feat).view(b, -1)
        w_td = self.global_pool(td_feat).view(b, -1)
        
        combined_stat = torch.cat([w_cop, w_sd, w_td], dim=1)
        
        # 💡 优化点 1：引入温度系数 T=2.0 软化 Softmax，避免非0即1的病态激进输出
        raw_logits = self.mlp(combined_stat) / 2.0 
        curr_weights = F.softmax(raw_logits, dim=-1)
        
        # 💡 优化点 2：一阶低通惯性演进 (Momentum = 0.7)
        # 当前步信任度由 30% 的当前瞬时统计量和 70% 的历史置信惯性共同决定
        if self.last_weights is None or self.last_weights.shape[0] != b:
            self.last_weights = curr_weights.clone().detach()
        else:
            if training_mode:
                curr_weights = 0.3 * curr_weights + 0.7 * self.last_weights
                self.last_weights = curr_weights.clone().detach()
                
        alpha = curr_weights[:, 0:1].view(b, 1, 1, 1)
        beta  = curr_weights[:, 1:2].view(b, 1, 1, 1)
        gamma = curr_weights[:, 2:3].view(b, 1, 1, 1)
        return alpha, beta, gamma


class TianmoucHNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        print("====== 正在加载 ConvNeXt-Tiny 预训练权重先验 ======")
        self.cop_net = timm.create_model('convnext_tiny', pretrained=True, features_only=True)
        for param in self.cop_net.parameters():
            param.requires_grad = False
            
        self.sd_net = ConvNeXt2StageSNN(inchannel=2, out_channels=384)
        self.td_net = ConvNeXt2StageSNN(inchannel=1, out_channels=384)
        
        self.quality_gate = SignalQualityGate(in_channels=384)
        
        self.hu_fuse = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=1, bias=False), 
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.register_buffer("current_feature_map", None, persistent=False)

    def reset_stream_state(self):
        self.current_feature_map = None
        self.quality_gate.last_weights = None # 🚀 核心优化：刷新航迹时务必重置权重历史
        self.sd_net.reset_state(history=False)
        self.td_net.reset_state(history=False)

    def get_oracle_rgb_feature(self, current_rgb_frame):
        with torch.no_grad(): 
            feats = self.cop_net(current_rgb_frame)
            oracle_feat = feats[2]
        return oracle_feat

    def forward(self, rgb_frame, sd_slice, td_slice, is_rgb_available=True):
        if is_rgb_available:
            feats = self.cop_net(rgb_frame)
            self.current_feature_map = feats[2].clone()  
            
        sd_feat = self.sd_net.step(sd_slice)  
        td_feat = self.td_net.step(td_slice)  
        
        if self.current_feature_map is None:
            self.current_feature_map = torch.zeros((sd_feat.shape[0], 384, 20, 40), device=sd_feat.device)
            
        sd_feat_aligned = F.interpolate(sd_feat, size=(20, 40), mode='nearest')
        td_feat_aligned = F.interpolate(td_feat, size=(20, 40), mode='nearest')
        
        # 传递 self.training 状态以维持流式验证的一致性
        alpha, beta, gamma = self.quality_gate(self.current_feature_map, sd_feat_aligned, td_feat_aligned, training_mode=self.training)
        
        dvs_weighted_fuse = beta * sd_feat_aligned + gamma * td_feat_aligned
        feature_delta = self.hu_fuse(dvs_weighted_fuse)
        
        self.current_feature_map = alpha * self.current_feature_map + feature_delta
        return self.current_feature_map


class TaskHead(nn.Module):
    """
    🚀 升级适配版预测 Head：完美接驳新特征，满血回归 [cx, cy, w, h] 四维检测框！
    """
    def __init__(self, in_channels=384, num_objects=3):
        super().__init__()
        self.num_objects = num_objects
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            # 🔒 核心修复：把 num_objects * 2 改为 num_objects * 4
            nn.Linear(256, num_objects * 4) 
        )
    def forward(self, feature_map):
        b = feature_map.shape[0]
        x = self.global_pool(feature_map).view(b, -1)
        out = self.fc(x).view(b, self.num_objects, 4) 
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