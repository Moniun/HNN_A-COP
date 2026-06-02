# dataset/tmdat_dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class TianmoucStreamingDataset(Dataset):
    """
    🚀 现代化全自闭环天眸流式检测数据集：
    自动检索规整化目录下的 seq_xxx 子文件，按需延迟加载，完美杜绝显存碎片与残留震荡。
    """
    def __init__(self, test=False, data_dir="/root/autodl-tmp/imagenet_vid_data/tianmouc_vid_proc") -> None:
        super().__init__()
        sub_folder = "val" if test else "train"
        self.target_dir = os.path.join(data_dir, sub_folder)
        
        # 检索当前目录下已经固化出来的样本标识指纹
        search_pattern = os.path.join(self.target_dir, "*_cop_t0.npy")
        cop_files = sorted(glob.glob(search_pattern))
        
        self.seq_ids = [os.path.basename(f).split('_cop_t0.npy')[0] for f in cop_files]
        if len(self.seq_ids) == 0:
            print(f"⚠️ Warning: 未在目录 '{self.target_dir}' 下检索到任何固化好的天眸特征矩阵，请先运行 vid_data_converter.py 仿真生成！")

    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, item):
        seq_id = self.seq_ids[item]
        
        # 延迟按需加载，极速释放内存空间
        cop = np.load(os.path.join(self.target_dir, f"{seq_id}_cop_t0.npy")) / 255.0 # [3, 320, 640]
        td  = np.load(os.path.join(self.target_dir, f"{seq_id}_aop_td.npy"))         # [1, 160, 160, 40]
        sd  = np.load(os.path.join(self.target_dir, f"{seq_id}_aop_sd.npy"))         # [2, 160, 160, 40]
        target_loc = np.load(os.path.join(self.target_dir, f"{seq_id}_label.npy"))   # [3, 4, 40]
        
        # 转换为张量交付下游
        cop_tensor = torch.from_numpy(cop).float()
        td_tensor  = torch.from_numpy(td).float()
        sd_tensor  = torch.from_numpy(sd).float()
        target_loc_tensor = torch.from_numpy(target_loc).float()
        
        # 兼容 main.py 内部需要的 cop_loc 刷新起点占位符
        cop_loc = target_loc_tensor[..., 0] 
        
        return cop_tensor, td_tensor, sd_tensor, cop_loc, target_loc_tensor