import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class TianmoucStreamingDataset(Dataset):
    """
    🚀 工业级标准化天眸流式检测通用 DataLoader (100% 解耦离线计算)
    """
    # 🔒 黄金路由线：默认直接切入我们刚刚固化好的 MOT17 时序特征大本营
    def __init__(self, test=False, data_dir="/root/autodl-tmp/HNN_A-COP/data/mot17_project/tianmouc_vid_proc") -> None:
        super().__init__()
        sub_folder = "val" if test else "train"
        self.target_dir = os.path.join(data_dir, sub_folder)
        
        # 稳健防线：如果测试时没有单独的 val，自动平滑路由到 train 序列跑通路
        if not os.path.exists(self.target_dir):
            self.target_dir = os.path.join(data_dir, "train")
        
        search_pattern = os.path.join(self.target_dir, "*_cop_t0.npy")
        cop_files = sorted(glob.glob(search_pattern))
        
        self.seq_ids = [os.path.basename(f).split('_cop_t0.npy')[0] for f in cop_files]
        if len(self.seq_ids) == 0:
            print(f"⚠️ 提示: 未在目录 '{self.target_dir}' 下检索到固化特征。如果是初次切换任务，请先运行对应的离线转换脚本！")

    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, item):
        seq_id = self.seq_ids[item]
        
        # 🔒 纯净搬运：无脑、极速加载落盘好的 4D/5D 原生天眸张量，CPU 零计算开销！
        cop = np.load(os.path.join(self.target_dir, f"{seq_id}_cop_t0.npy"))        # [3, 320, 640, 40]
        td  = np.load(os.path.join(self.target_dir, f"{seq_id}_aop_td.npy"))         # [1, 160, 160, 40]
        sd  = np.load(os.path.join(self.target_dir, f"{seq_id}_aop_sd.npy"))         # [2, 160, 160, 40]
        target_loc = np.load(os.path.join(self.target_dir, f"{seq_id}_label.npy"))   # [3, 4, 40]
        
        cop_tensor = torch.from_numpy(cop).float()
        td_tensor  = torch.from_numpy(td).float()
        sd_tensor  = torch.from_numpy(sd).float()
        target_loc_tensor = torch.from_numpy(target_loc).float()
        
        cop_loc = target_loc_tensor[..., 0] 
        return cop_tensor, td_tensor, sd_tensor, cop_loc, target_loc_tensor