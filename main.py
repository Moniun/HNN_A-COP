from dataset.tmdat_dataset import TianmoucStreamingDataset
from models import TianmoucHNNBackbone, TaskHead, FullModel
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import time


def train():
    epoch_num = 100
    save_period = 1
    load_ckpt_path = "ckpt/HNN_backbone.ckpt" # 默认点火载入你刚刚做完自监督训练的 Backbone 权重
    save_ckpt_path = "ckpt/HNN_detection_head.ckpt"
    Path(os.path.dirname(save_ckpt_path)).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter('summary/task_head/train_predictor_{}'.format(int(time.time())))
    
    backbone = TianmoucHNNBackbone().cuda()
    # 🚀 核心适配修改：下游检测 Head 接收端通道数无损更新为标准的 384 维
    task_head = TaskHead(in_channels=384, num_objects=3).cuda()
    
    if load_ckpt_path and os.path.exists(load_ckpt_path):
        state_dict = torch.load(load_ckpt_path, map_location=torch.device("cuda:0"))
        backbone.load_state_dict(state_dict.get('backbone', {}), strict=False)
        print(f"====== 成功无缝加载预训练好的现代化大核 Backbone 权重先验 ======")
    
    for param in backbone.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(task_head.parameters()), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[400], gamma=0.1)
    criterion = nn.MSELoss()
    
    train_data = DataLoader(TianmoucStreamingDataset(test=False), batch_size=4, shuffle=True, num_workers=4)
    
    for epoch in tqdm(range(epoch_num)):
        for step, data in enumerate(train_data):
            cop, td, sd, cop_loc, target_loc = data
            cop = cop.cuda()
            td = td.cuda()
            sd = sd.cuda()
            target_loc = target_loc.cuda().float()
            
            backbone.reset_stream_state()
            total_loss = 0
            
            T_steps = td.shape[-1]
            # === 找到 train() 内部循环 ===
            for t in range(T_steps):
                is_rgb_available = (t == 0)

                # 🔒 核心修复：输入给 backbone 的 RGB 图像按当前时间片解包
                current_rgb_frame = cop[..., t] if is_rgb_available else cop[..., 0] 

                current_feat = backbone(
                    rgb_frame=current_rgb_frame, 
                    sd_slice=sd[..., t], 
                    td_slice=td[..., t], 
                    is_rgb_available=is_rgb_available
                )
                
                pred_loc = task_head(current_feat)
                loss_step = criterion(pred_loc, target_loc[..., t])
                total_loss += loss_step
            
            total_loss = total_loss / T_steps
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(task_head.parameters()), 1)
            optimizer.step()

            writer.add_scalar('training loss', total_loss.item(), step + 1 + epoch * len(train_data))

        scheduler.step()
        if (epoch + 1) % save_period == 0:
            print("\rsaving at epoch {}, loss={:.3f}, path: {}".format(epoch + 1, total_loss.item(), save_ckpt_path))
            torch.save({
                'backbone': backbone.state_dict(),
                'head': task_head.state_dict()
            }, save_ckpt_path)

def test():
    # 1. 搭建拓扑架构毛坯房
    backbone = TianmoucHNNBackbone().cuda()
    task_head = TaskHead(in_channels=384, num_objects=3).cuda()

    # 🔒 黄金加载线：精确定位你在 train() 完固化下来的满血成果权重包
    trained_ckpt_path = "ckpt/HNN_detection_head.ckpt"
    
    if os.path.exists(trained_ckpt_path):
        print(f"====== 📡 正在强行注入微调完备的 HNN 满血目标检测参数... ======")
        checkpoint = torch.load(trained_ckpt_path, map_location=torch.device("cuda:0"))
        
        # 🔒 核心修复：分别将训练好的 Backbone 和 4通道 TaskHead 权重无损全量恢复！
        backbone.load_state_dict(checkpoint.get('backbone', {}), strict=True)
        task_head.load_state_dict(checkpoint.get('head', {}), strict=True)
        print(f"====== 🎉 [两路权重合体成功] 正在带全量大脑先验切入推理评估通路 ======")
    else:
        print(f"⚠️ 警报：在 '{trained_ckpt_path}' 下未找到微调权重！将使用随机权重进行跑通测试。")

    # 进入标准的评估模式（锁定 BatchNorm 以及一阶低通信号质量门控的统计量演进）
    backbone.eval()
    task_head.eval()

    test_data = DataLoader(TianmoucStreamingDataset(test=True), batch_size=1, pin_memory=True, shuffle=False)
    video = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 320)) 

    print("🎬 正在启动全时序动态图像渲染测试...")
    
    # 强制在不记录计算图的模式下进行流式推理，极大压缩显存开销
    with torch.no_grad():
        for step, data in enumerate(test_data):
            cop, td, sd, cop_loc, target_loc = data
            cop = cop.cuda()
            td = td.cuda()
            sd = sd.cuda()

            backbone.reset_stream_state()
            T_steps = td.shape[-1]

            pred_loc_list = []
            for t in range(T_steps):
                # 🔒 测试时背景和物体同步跟随时间轴平移，全时步放开特征融合演进
                is_rgb_available = True 
                
                current_feat = backbone(
                    rgb_frame=cop[..., t], 
                    sd_slice=sd[..., t], 
                    td_slice=td[..., t], 
                    is_rgb_available=is_rgb_available
                )
                pred_loc = task_head(current_feat)
                pred_loc_list.append(pred_loc.squeeze().data.cpu().numpy())

            pred_loc_arr = np.stack(pred_loc_list, axis=-1).astype(np.float32)
            gt_loc = target_loc.squeeze().numpy().astype(np.float32)

            cop_np_seq = cop.squeeze().data.cpu().numpy() # [3, 320, 640, 40]

            for t in range(T_steps):
                plt.gca().clear()

                # 动态捕获当前时间步平移后的真实图像背景
                frame_t = cop_np_seq[..., t]
                if frame_t.max() <= 1.01:
                    frame_t = frame_t * 255.0
                frame_t = np.clip(frame_t, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(frame_t.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

                for o in range(3):
                    cx, cy, w, h = pred_loc_arr[o, :, t]
                    gcx, gcy, gw, gh = gt_loc[o, :, t]

                    # 解算红色预测框的左上角与右下角
                    px1, py1 = int(cx - w / 2), int(cy - h / 2)
                    px2, py2 = int(cx + w / 2), int(cy + h / 2)
                    
                    # 解算绿色真值框的左上角与右下角
                    gx1, gy1 = int(gcx - gw / 2), int(gcy - gh / 2)
                    gx2, gy2 = int(gcx + gw / 2), int(gcy + gh / 2)

                    # 严格、独立地对所有画框坐标执行越界裁剪，粉碎原有的笔误
                    px1, px2 = np.clip([px1, px2], 0, 640 - 1)
                    py1, py2 = np.clip([py1, py2], 0, 320 - 1)
                    gx1, gx2 = np.clip([gx1, gx2], 0, 640 - 1)
                    gy1, gy2 = np.clip([gy1, gy2], 0, 320 - 1)

                    # 🟥 绘制红色预测框
                    img = cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2) 
                    # 🟩 绘制绿色真值框
                    img = cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2) 

                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.pause(0.01)
                video.write(img)

    video.release()
    print("📊 视频渲染完毕，成果已安全导出至项目根目录下的 demo.mp4！")


if __name__ == "__main__":
    train()
    # test()