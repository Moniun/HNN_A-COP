# /root/autodl-tmp/HNN_A-COP/eval.py
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset.tmdat_dataset import TianmoucStreamingDataset
from models import TianmoucHNNBackbone, TaskHead

def eval():
    # 1. 搭建拓扑架构
    backbone = TianmoucHNNBackbone().cuda()
    task_head = TaskHead(in_channels=384, num_objects=3).cuda()

    # 🔒 载入训练成果全量参数
    trained_ckpt_path = "ckpt/HNN_detection_head.ckpt"
    if os.path.exists(trained_ckpt_path):
        print(f"====== 📡 正在强行注入微调完备的 HNN 满血目标检测参数... ======")
        checkpoint = torch.load(trained_ckpt_path, map_location=torch.device("cuda:0"))
        backbone.load_state_dict(checkpoint.get('backbone', {}), strict=True)
        task_head.load_state_dict(checkpoint.get('head', {}), strict=True)
        print(f"====== 🎉 [两路大脑权重合体成功] 正在跨入官方纯净测试集流式前向评估通路 ======")
    else:
        print(f"⚠️ 警报：在 '{trained_ckpt_path}' 下未找到微调权重！将使用随机权重进行跑通测试。")

    # 全线卡死推理评估模式
    backbone.eval()
    task_head.eval()

    # 2. 强制拉取真正的官方无标签测试集（test=True）
    dataset = TianmoucStreamingDataset(test=True)
    test_data = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False)
    
    # 建立视频导出通道，帧率设为 15 FPS
    video = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 320)) 

    print("🎬 正在启动官方长视频流全时序物理因果状态连续推理测试...")
    
    # 🔒 核心动力学长线记忆指针：记录上一个分块所属的视频主名
    last_video_name = None

    # 强制在不记录计算图的模式下运行，极大压缩显存与算力开销
    with torch.no_grad():
        for step, data in enumerate(test_data):
            cop, td, sd, cop_loc, target_loc = data
            cop, td, sd = cop.cuda(), td.cuda(), sd.cuda()

            # 🔒 物理防线：识别当前片段所属的真实长视频序列主名
            # 例如 "MOT17-02_seg_01" -> 提纯出大视频主名 "MOT17-02"
            current_seq_id = dataset.seq_ids[step]
            current_video_name = current_seq_id.split('_seg_')[0]

            # 🔒 只有更换全新视频序列时，才执行全盘重置！
            # 处于同一个长视频的不同分块内部时，SNN 膜电位和信号质量门控惯性 100% 连续演进，还原真实芯片物理常开状态！
            if current_video_name != last_video_name:
                print(f"📡 [场景大阶跃点火] 检测到进入全新官方测试场景: {current_video_name}，脉冲状态流归零初始化。")
                backbone.reset_stream_state()
                last_video_name = current_video_name
            else:
                # 同一长视频内部，仅在每个分块的第一帧触发空间主路吃入一次 RGB 更新特征缓存
                # AOP 脉冲通路通过 backbone 内部的 .step() 机制在片段边界实现 100% 物理常开演进！
                pass

            T_steps = td.shape[-1]
            pred_loc_list = []
            
            for t in range(T_steps):
                is_rgb_available = (t == 0)
                
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
            cop_np_seq = cop.squeeze().data.cpu().numpy()

            # 3. 实时画面渲染
            for t in range(T_steps):
                plt.gca().clear()
                frame_t = cop_np_seq[..., t]
                if frame_t.max() <= 1.01:
                    frame_t = frame_t * 255.0
                frame_t = np.clip(frame_t, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(frame_t.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

                for o in range(3):
                    # 从网络中吐出来的是 [0, 1] 相对值，在此原地乘以画布物理尺寸还原
                    cx = pred_loc_arr[o, 0, t] * 640.0
                    cy = pred_loc_arr[o, 1, t] * 320.0
                    w  = pred_loc_arr[o, 2, t] * 640.0
                    h  = pred_loc_arr[o, 3, t] * 320.0

                    gcx = gt_loc[o, 0, t] * 640.0
                    gcy = gt_loc[o, 1, t] * 320.0
                    gw  = gt_loc[o, 2, t] * 640.0
                    gh  = gt_loc[o, 3, t] * 320.0

                    # 解算红色预测框的左上角与右下角
                    px1, py1 = int(cx - w / 2), int(cy - h / 2)
                    px2, py2 = int(cx + w / 2), int(cy + h / 2)
                    
                    # 解算绿色真值框的左上角与右下角
                    gx1, gy1 = int(gcx - gw / 2), int(gcy - gh / 2)
                    gx2, gy2 = int(gcx + gw / 2), int(gcy + gh / 2)

                    # 画框边界严格保护
                    px1, px2 = np.clip([px1, px2], 0, 640 - 1)
                    py1, py2 = np.clip([py1, py2], 0, 320 - 1)
                    gx1, gx2 = np.clip([gx1, gx2], 0, 640 - 1)
                    gy1, gy2 = np.clip([gy1, gy2], 0, 320 - 1)

                    # 🟥 绘制红色 HNN 脉冲流预测框 (在全新测试场景下同步高灵敏追踪闪烁)
                    img = cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2) 
                    
                    # 🟩 绘制绿色真值框 (注意：官方测试集由于没有真值，会静静地停留在 [0,0,0,0] 隐藏，属于正常现象)
                    if gcx > 0 or gcy > 0:
                        img = cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2) 

                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.pause(0.01)
                video.write(img)

    video.release()
    print("📊 成果大片渲染完毕，全序列因果追踪成果已安全导出至项目根目录下的 demo.mp4！")

if __name__ == "__main__":
    eval()