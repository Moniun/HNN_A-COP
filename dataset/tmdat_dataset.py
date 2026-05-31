import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tianmoucv.data.tianmoucData import TianmoucDataReader
from tianmoucv.proc.denoise import denoise_defualt_args


class TianmoucStreamingDataset(Dataset):
    """
    通用天眸芯流式感知数据集
    去除了转盘追踪任务的硬编码逻辑，纯粹作为通用的分布式多模态特征与时序全量标签的供给器
    """
    def __init__(self, test=False, data_dir="tianmouc_data") -> None:
        super().__init__()
        self.data_dir = data_dir
        self.is_test = test
        prefix = "test_" if test else ""
        
        # 数据存放在 proc_npy 子文件夹下
        proc_dir = os.path.join(data_dir, "proc_npy")
        
        # 1. 载入离线持久化好的高保真 numpy 矩阵
        self.frames = np.load(abspath(f"{proc_dir}/{prefix}cop_frame.npy"))     # 静态认知帧 [N, 3, 320, 640]
        self.td_events = np.load(abspath(f"{proc_dir}/{prefix}aop_td.npy"))   # 时间差分流 [N, 1, 160, 160, T]
        self.sd_events = np.load(abspath(f"{proc_dir}/{prefix}aop_sd.npy"))   # 空间差分流 [N, 2, 160, 160, T]

        # 确保通道轴在第 1 维 [B, C, H, W]
        if len(self.frames.shape) == 4 and self.frames.shape[-1] == 3:
            self.frames = self.frames.transpose(0, 3, 1, 2)

    def __len__(self):
        return self.frames.shape[0]
    
    def __getitem__(self, item):
        cop = self.frames[item]       
        td = self.td_events[item]     
        sd = self.sd_events[item]     
        T_steps = td.shape[-1]
        
        target_loc = self.load_generic_sequence_labels(item, T_steps, test=self.is_test)
        
        cop_loc = target_loc[..., 0] 
        return cop, td, sd, cop_loc, target_loc

    def load_generic_sequence_labels(self, item, T_steps, test=False):
        """
        根据 test 状态，自动去对应的文件夹读取训练标签或测试标签
        """
        prefix = "test_" if test else ""
        
        label_path = abspath(f"{self.data_dir}/labels/{prefix}{item}.npy")
        
        if os.path.exists(label_path):
            return np.load(label_path)
        else:
            print(f"⚠️ Warning: Label file {label_path} not found! Returning zero placeholders.")
            return np.zeros((3, 2, T_steps), dtype=np.float32)


def temporal_accumulate(data, target_ts):
    """
    流式核心算子：将芯片原生的 T_raw 个高频时间片，通过时域邻域滑窗叠加（Sum），
    智能无损压缩为你人为指定的 target_ts 个前向传播时间微步。
    输入 data 形状: [Channel, Time, Height, Width]
    """
    C, T, H, W = data.shape
    if target_ts is None or target_ts == T:
        return data
    
    indices_groups = np.array_split(np.arange(T), target_ts)
    accumulated_steps = []
    
    for group in indices_groups:
        step_sum = np.sum(data[:, group, :, :], axis=1)
        accumulated_steps.append(step_sum)
        
    return np.stack(accumulated_steps, axis=1)


def tmdat_2_numpy(tmdat_dir, output_dir="tianmouc_data", target_ts=None, is_test=False):
    """
    离线数据转换器：读取原始硬件压缩包 .tmdat 并导出为适配 HNN 独立通路的 Numpy 矩阵
    
    参数:
        tmdat_dir (str): 存放原始 .tmdat 文件的根目录
        output_dir (str): 导出 .npy 特征阵列的根目录
        target_ts (int): 目标时域对齐微步数。若为 None 则完全保留硬件原生高频切片数
        is_test (bool): 是否为测试集。如果是 True，导出的文件名会自动带上 "test_" 前缀
    """
    # 数据导出到 proc_npy 子文件夹
    proc_dir = os.path.join(output_dir, "proc_npy")
    os.makedirs(proc_dir, exist_ok=True)
    d_args = denoise_defualt_args()
    
    reader = TianmoucDataReader(path=tmdat_dir, N=1, aop_denoise=True, aop_denoise_args=d_args, training=True, use_data_parser=False)
    
    frames_data, td_data, sd_data = [], [], []
    
    prefix = "test_" if is_test else ""
    
    print(f"开始解析原生天眸芯 {'【测试集】' if is_test else '【训练集】'} .tmdat 数据流，总片段数: {len(reader)}")
    for idx in range(len(reader)):
        sample = reader[idx]
        if sample is None:
            continue
            
        cop_tensor = sample['F0'] 
        cop_np = cop_tensor.numpy()
        if cop_np.shape[-1] == 3:
            cop_np = cop_np.transpose(2, 0, 1)
        frames_data.append(cop_np)
        
        raw_aop = sample['rawDiff'].numpy() 
        if target_ts is not None:
            raw_aop = temporal_accumulate(raw_aop, target_ts)
            
        td_raw = raw_aop[0:1, ...]
        sd_raw = raw_aop[1:3, ...]
        
        td_data.append(td_raw.transpose(0, 2, 3, 1))
        sd_data.append(sd_raw.transpose(0, 2, 3, 1))
        
    np.save(os.path.join(proc_dir, f"{prefix}cop_frame.npy"), np.array(frames_data, dtype=np.float32))
    np.save(os.path.join(proc_dir, f"{prefix}aop_td.npy"), np.array(td_data, dtype=np.float32))
    np.save(os.path.join(proc_dir, f"{prefix}aop_sd.npy"), np.array(sd_data, dtype=np.float32))
    print(f"转换完成！HNN 通路的独立特征已被固化导出至: {proc_dir}/{prefix}*.npy")


def abspath(path):
    return os.path.join(os.path.dirname(__file__), path)


if __name__ == "__main__":
    output_data_root = abspath("tianmouc_data")
    
    # train_tmdat_dir = abspath("raw_tmdat/train") 
    # tmdat_2_numpy(train_tmdat_dir, output_dir=output_data_root, target_ts=None, is_test=False)
    
    # test_tmdat_dir = abspath("raw_tmdat/test") 
    # tmdat_2_numpy(test_tmdat_dir, output_dir=output_data_root, target_ts=None, is_test=True)

    train_tmdat_dir = abspath("/root/autodl-tmp/HNN_A-COP/tianmouc_data/driving_subset/NM_outdoor_cross_6") 
    tmdat_2_numpy(train_tmdat_dir, output_dir=output_data_root, target_ts=None, is_test=False)