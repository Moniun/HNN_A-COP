import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# 导入天眸c官方提供的解码与数据读取模块
from tianmoucv.data.tianmoucData import TianmoucDataReader
from tianmoucv.proc.denoise import denoise_defualt_args


class TurningDiskDataset(Dataset):
    def __init__(self, test=False) -> None:
        super().__init__()
        # 预处理后保存的天眸c原生高保真 numpy 矩阵路径
        if not test:
            self.frames = np.load(abspath("tianmouc_data/cop_frame.npy"))     # 原生静态认知帧 [N, 3, 320, 640]
            self.td_events = np.load(abspath("tianmouc_data/aop_td.npy"))   # 原生时间差分流 [N, 1, 160, 160, T]
            self.sd_events = np.load(abspath("tianmouc_data/aop_sd.npy"))   # 原生空间差分流 [N, 2, 160, 160, T]
        else:
            self.frames = np.load(abspath("tianmouc_data/test_cop_frame.npy"))
            self.td_events = np.load(abspath("tianmouc_data/test_aop_td.npy"))
            self.sd_events = np.load(abspath("tianmouc_data/test_aop_sd.npy"))

        if len(self.frames.shape) == 4 and self.frames.shape[-1] == 3:
            self.frames = self.frames.transpose(0, 3, 1, 2)

    def __len__(self):
        return self.frames.shape[0]
    
    def __getitem__(self, item):
        cop = self.frames[item]       # 形状: [3, 320, 640] (天眸c原生RGB分辨率)
        td = self.td_events[item]     # 形状: [1, 160, 160, T] (天眸c原生AOP分辨率)
        sd = self.sd_events[item]     # 形状: [2, 160, 160, T] (天眸c原生AOP分辨率)
        
        # 1. 提取COP帧中的多目标位置真值标签 [Num_Objects, 2]
        # 传入 get_target 的图像形状转换为 [H, W, C] 以匹配 OpenCV 接口
        cop_loc = self.get_target(cop.transpose(1, 2, 0), isFrame=True)
        
        # 2. 逐时间步提取TD事件中的多目标运动轨迹真值 [Num_Objects, 2, T]
        # get_target 内部会自适应 td 此时的 [160, 160] 尺寸进行质心归一化
        target_loc = np.stack([
            self.get_target(td[:, ..., t].transpose(1, 2, 0), isFrame=False) 
            for t in range(td.shape[-1])
        ], -1)
        
        # 【完美对接网络接口】返回5个元素，其空间长宽和时间步完全保持原生或叠加后的状态
        return cop, td, sd, cop_loc, target_loc

    @staticmethod
    def get_target(img, isFrame=True):
        """
        自适应输入图像尺寸的多目标质心标签自动提取算法
        """
        if isFrame:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            # 天眸c 官方默认输出为 3 通道 RGB，需先转为单通道灰度图再做二值化
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            img_bin = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        else:
            # 脉冲流输入：若有多通道（如SDL/SDR）则取最大绝对值压缩为 2D 轮廓
            if len(img.shape) == 3:
                img_2d = np.max(np.abs(img), axis=2)
            else:
                img_2d = np.abs(img)
            img_bin = (img_2d > 0).astype(np.uint8)

        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        area_sort = np.argsort([cv2.contourArea(c) for c in contours])
        area_sort = area_sort[-2:-5:-1] if isFrame else area_sort[:-4:-1]
        
        valid_idx = [i for i in area_sort if i < len(contours)]
        if len(valid_idx) == 0:
            return np.zeros((3, 2))  # 防御边界：返回 3 个目标的全零占位
            
        contours = np.asarray(contours, dtype=object)[valid_idx]

        x, y, w, h = np.array([cv2.boundingRect(cnt) for cnt in contours]).T
        xc, yc = x + w / 2, y + h / 2
        center = np.stack([xc, yc], -1)
        
        # 自适应当前输入的宽和高进行质心排序
        r = ((img_bin.shape[1] / 2 - xc) ** 2 + (img_bin.shape[0] / 2 - yc) ** 2) ** 0.5
        center = center[np.argsort(r)]

        # 归一化中心坐标 (完全依据当前 img_bin 的物理长宽自适应)
        center = center - np.array(img_bin.shape[:2])[np.newaxis, ::-1] / 2 + 0.5

        # 规范多目标数量输出为 3，确保 Batch 稳定
        if center.shape[0] < 3:
            pad = np.zeros((3 - center.shape[0], 2))
            center = np.concatenate([center, pad], axis=0)
        else:
            center = center[:3, :]
            
        return center


def temporal_accumulate(data, target_ts):
    """
    核心算子：将原始的 T 个时间步，通过时域邻域叠加（Sum），无损压缩为目标 target_ts 个时间步
    输入 data 形状: [Channel, Time, Height, Width]
    """
    C, T, H, W = data.shape
    if target_ts is None or target_ts == T:
        return data
    
    # 利用 np.array_split 智能应对无法整除的情况（例如 25 步切成 6 步）
    indices_groups = np.array_split(np.arange(T), target_ts)
    accumulated_steps = []
    
    for group in indices_groups:
        # 将属于当前时间窗口内的所有原始脉冲在时域上进行累加（Sum）
        step_sum = np.sum(data[:, group, :, :], axis=1) # 消除组内时间轴 -> [Channel, Height, Width]
        accumulated_steps.append(step_sum)
        
    # 重新在时间轴（dim=1）上拼接
    return np.stack(accumulated_steps, axis=1) # 返回 [Channel, target_ts, Height, Width]


def tmdat_2_numpy(tmdat_dir, output_dir="tianmouc_data", target_ts=None):
    """
    离线读取原始 .tmdat 数据包，支持原生时间步保留或人为指定时间步叠加
    """
    os.makedirs(output_dir, exist_ok=True)
    
    d_args = denoise_defualt_args()
    # 实例化官方 DataReader，内部自动挂载 C++ 编写的 rod_decoder 并进行降噪
    reader = TianmoucDataReader(path=tmdat_dir, N=1, aop_denoise=True, aop_denoise_args=d_args, training=True, use_data_parser=False)
    
    frames_data = []
    td_data = []
    sd_data = []
    
    print(f"正在读取并解析 .tmdat 原始数据集，总样本数: {len(reader)} ...")
    for idx in range(len(reader)):
        sample = reader[idx]
        if sample is None:
            continue
            
        # 1. 处理 COP 认知图像帧 -> 直接保留官方原始输出的 [3, 320, 640] 形状，不进行任何 Resize
        cop_tensor = sample['F0'] 
        cop_np = cop_tensor.numpy()
        
        if cop_np.shape[-1] == 3:
            cop_np = cop_np.transpose(2, 0, 1)
            
        frames_data.append(cop_np)
        # frames_data.append(cop_tensor.numpy()) # [3, 320, 640]
        
        # 2. 处理 AOP 差分脉冲时空流 -> 官方默认输出原始形状为 [3, itter, 160, 160]
        raw_aop = sample['rawDiff'].numpy() 
        
        # 💡 调用时域叠加机制：实现人为定义时间步数与内生时间步的完美兼容
        if target_ts is not None:
            raw_aop = temporal_accumulate(raw_aop, target_ts)
            
        # 3. 通道剥离与流分流
        td_raw = raw_aop[0:1, ...] # 时间差分 TD 形状: [1, T_fused, 160, 160]
        sd_raw = raw_aop[1:3, ...] # 空间差分 SD 形状: [2, T_fused, 160, 160]
        
        # 4. 维度置换（Permute）：将时间轴 T 移到最后一维，完全适配你的双通路 SNN 输入标准
        td_fused = td_raw.transpose(0, 2, 3, 1) # [1, 160, 160, T_fused]
        sd_fused = sd_raw.transpose(0, 2, 3, 1) # [2, 160, 160, T_fused]
        
        td_data.append(td_fused)
        sd_data.append(sd_fused)
        
    # 执行全量持久化保存
    np.save(os.path.join(output_dir, "cop_frame.npy"), np.array(frames_data, dtype=np.float32))
    np.save(os.path.join(output_dir, "aop_td.npy"), np.array(td_data, dtype=np.float32))
    np.save(os.path.join(output_dir, "aop_sd.npy"), np.array(sd_data, dtype=np.float32))
    print(f"🎉 转换成功！已完全保留天眸c原始长宽。数据矩阵已导出至: {output_dir}")


def abspath(path):
    return os.path.join(os.path.dirname(__file__), path)


if __name__ == "__main__":
    tmdat_data_root = abspath("tianmouc_data")
    
    # 💡 使用方法说明：
    # 1. 若你想【100%自动保留硬件原始切片个数】，将 target_ts 设为 None 即可：
    tmdat_2_numpy(tmdat_data_root, target_ts=None)
    
    # 2. 若你想【人为指定切片个数】，比如强行压缩为 5 个时间步，系统会自动在内部进行时域邻域脉冲叠加：
    # tmdat_2_numpy(tmdat_data_root, target_ts=5)