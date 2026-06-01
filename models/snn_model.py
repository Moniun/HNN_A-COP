import torch
import torch.nn as nn
import numpy as np

update_v = 'default'


class ActFun(torch.autograd.Function):
    """A Heaviside step function that is made differentiable with surrogate gradient"""
    lens = 0.5

    @staticmethod
    def forward(ctx, _input):
        ctx.save_for_backward(_input)
        return _input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        _input, = ctx.saved_tensors
        return grad_output * _input.abs().lt(ActFun.lens).float()


class SNNLayer(nn.Module):
    """
    🔒 100% 完好继承原版 SNN 物理属性与控制参数
    """
    def __init__(self, layer, bn=True, thresh=None, thresh_grad=True, decay=0.0, decay_grad=False, bypass_in=False, update_v=update_v):
        super(SNNLayer, self).__init__()
        self.layer = layer
        self.state = [0., 0.]  # [mem, spike]

        if thresh is None:
            thresh = 0.0 if update_v == 'rnn' else 0.5

        self.thresh = nn.Parameter(torch.ones((1, layer.out_channels, 1, 1)) * thresh, requires_grad=thresh_grad)
        self.decay = nn.Parameter(torch.ones((1, layer.out_channels, 1, 1)) * decay, requires_grad=decay_grad)

        self.bn = nn.BatchNorm2d(layer.out_channels) if bn else None
        self.bypass_bn = nn.BatchNorm2d(layer.out_channels) if bn and bypass_in else None

        if bn and thresh:
            self.bn.weight.data = self.thresh.data.view(-1) / (2**0.5 if bypass_in else 1)
            if bypass_in:
                self.bypass_bn.weight.data = self.thresh.data.view(-1) / 2**0.5

        self.act_func = nn.ReLU(inplace=False) if update_v == 'rnn' else ActFun.apply
        self.update_v = update_v

    def update_state(self, x, bypass_in):
        layer_in = self.bn(self.layer(x)) if self.bn is not None else self.layer(x)
        if bypass_in is not None:
            layer_in += self.bypass_bn(bypass_in) if self.bn is not None else bypass_in

        if self.update_v == 'default':
            self.state[0] = self.state[0] * (1. - self.state[1]) * self.decay + layer_in
        elif self.update_v == 'bursting':
            self.state[0] = self.state[0] * self.decay - self.state[1] * self.thresh + layer_in
        elif self.update_v == 'rnn':
            self.state[0] = self.state[0] * self.decay + layer_in

        self.state[1] = self.act_func(self.state[0] - self.thresh)

    def reset_state(self, history):
        self.state = [self.state[0].detach(), self.state[1].detach()] if history else [0., 0.]
        self.decay.data = self.decay.clamp(min=0., max=1.).data

    def forward(self, x, bypass_in=None):
        self.update_state(x, bypass_in)
        return self.state[1]


class ConvNeXtBlockSNN(nn.Module):
    """
    🚀 拓扑现代化改造：利用原版 SNNLayer 拼装成的 ConvNeXt 风格脉冲模块
    （深度空间大核 $7\times7$ + 逆残差 Inverted Bottleneck + 完美承接类脑不应期机制）
    """
    def __init__(self, dim):
        super().__init__()
        # 1. 深度大核脉冲卷积层 (groups=dim, kernel=7) -> 建立宏观时空视场
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.snn_dw = SNNLayer(self.dwconv, bn=True)
        
        # 2. 逆残差通道放大层 (通道放大 4 倍)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=False)
        self.snn_pw1 = SNNLayer(self.pwconv1, bn=True)
        
        # 3. 通道还原层 + 🔒 完美对齐并挂载原版神经元特有的真实物理残差机制（bypass_in=True）
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=False)
        self.snn_residual = SNNLayer(self.pwconv2, bn=True, bypass_in=True)

    def reset_state(self, history):
        self.snn_dw.reset_state(history)
        self.snn_pw1.reset_state(history)
        self.snn_residual.reset_state(history)

    def forward(self, x):
        # 记录主路残差
        residual = x
        out = self.snn_dw(x)
        out = self.snn_pw1(out)
        # 🔒 将残差包裹在类脑神经动力学机制内安全完成不应期叠加
        out = self.snn_residual(out, bypass_in=residual)
        return out


class ConvNeXt2StageSNN(nn.Module):
    """
    🚀 拓扑现代化改造：全面无缝适配 ConvNeXt-Tiny 的两阶段等比例各向同性脉冲主干
    """
    def __init__(self, inchannel=2, out_channels=384):
        super().__init__()
        
        # Stem 下采样：吃进 160x160 的原生平面脉冲，通过 4x4 大卷积直接以无损格平铺方式降维至 40x40
        self.stem = SNNLayer(nn.Conv2d(inchannel, 96, kernel_size=4, stride=4, padding=0, bias=False), bn=True)
        
        # Stage 1 拓扑演进：降维至 20x20
        self.stage1_down = SNNLayer(nn.Conv2d(96, 192, kernel_size=2, stride=2, padding=0, bias=False), bn=True)
        self.stage1_blocks = nn.ModuleList([ConvNeXtBlockSNN(dim=192) for _ in range(2)])
        
        # Stage 2 拓扑演进：保持 20x20，通道升至 384 维，与大模型老师形成完美物理各向同性
        self.stage2_down = SNNLayer(nn.Conv2d(192, out_channels, kernel_size=1, stride=1, padding=0, bias=False), bn=True)
        self.stage2_blocks = nn.ModuleList([ConvNeXtBlockSNN(dim=out_channels) for _ in range(2)])
        
        # 🔒 4. 终极留存核对：100% 原汁原味保留 RNN 机制的最后一层，不发火、不截断，提取纯粹的 state[0] 膜电位！
        self.final_rnn_layer = SNNLayer(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            decay=0., decay_grad=False, update_v='rnn', bn=False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def reset_state(self, history=False):
        self.stem.reset_state(history)
        self.stage1_down.reset_state(history)
        for block in self.stage1_blocks: block.reset_state(history)
        self.stage2_down.reset_state(history)
        for block in self.stage2_blocks: block.reset_state(history)
        self.final_rnn_layer.reset_state(history)

    def step(self, x):
        """处理单步时间片的输入 x: [B, C, H, W]"""
        out = self.stem(x)
        
        out = self.stage1_down(out)
        for block in self.stage1_blocks: out = block(out)
            
        out = self.stage2_down(out)
        for block in self.stage2_blocks: out = block(out)
            
        # 🔒 5. 原版特异性特写：驱使 RNN 通路演进
        _ = self.final_rnn_layer(out)
        # 🔒 稳健提取自回归循环最看重的未硬截断的连续膜电位（Voltage）
        out_mem = self.final_rnn_layer.state[0]
        return out_mem

    def forward(self, net_in):
        self.reset_state()
        out_list = []
        for t in range(net_in.shape[-1]):
            net_out = self.step(net_in[..., t])
            out_list.append(net_out)
        return torch.stack(out_list, -1)