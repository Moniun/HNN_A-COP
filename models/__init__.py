from .ann_model import Bottleneck
from .snn_model import ConvNeXt2StageSNN
from .full_model import TianmoucHNNBackbone, TaskHead, FullModel

DetectionHead = TaskHead

__all__ = [
    'Bottleneck',
    'ConvNeXt2StageSNN',
    'TianmoucHNNBackbone',
    'TaskHead',
    'DetectionHead',
    'FullModel'
]
