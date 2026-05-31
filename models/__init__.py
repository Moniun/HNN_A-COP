from .ann_model import Bottleneck, ResNet2Stage
from .snn_model import ResNet2StageSNN
from .full_model import TianmoucHNNBackbone, TaskHead, FullModel

DetectionHead = TaskHead

__all__ = [
    'Bottleneck',
    'ResNet2Stage',
    'ResNet2StageSNN',
    'TianmoucHNNBackbone',
    'TaskHead',
    'DetectionHead',
    'FullModel'
]
