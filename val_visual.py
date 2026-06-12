# /root/autodl-tmp/HNN_A-COP/val_visual.py
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset.tmdat_dataset import TianmoucStreamingDataset
from models import TianmoucHNNBackbone, TaskHead

def run_val_visual():
    # 1. 搭建相同的网络拓扑
    backbone = TianmoucHNNBackbone().cuda()
    task_head = TaskHead(in_channels=384, num_objects=3).cuda()

    # 🔒 强行注入你刚刚微调收敛完备的满血权重
    trained_ckpt_path = "ckpt/HNN_detection_head.ckpt"
    if os.path.exists(trained_ckpt_path):
        print(f"====== 📡 正在注入收敛至 0.02 级别的 HNN 满血检测参数... ======")
        checkpoint = torch.load(trained_ckpt_path, map_location=torch.device("cuda:0"))
        backbone.load_state_dict(checkpoint.get('backbone', {}), strict=True)
        task_head.load_state_dict(checkpoint.get('head', {}), strict=True)
        print(f"====== 🎉 [两路大脑权重合体成功] 正在切入【训练集】时空同步可视化渲染通路 ======")
    else:
        print(f"⚠️ 警报：在 '{trained_ckpt_path}' 下未找到微调权重！将使用随机权重进行跑通测试。")

    # 卡死推理评估模式
    backbone.eval()
    task_head.eval()

    # 2. 🔒 核心改动：强制拉取包含真实 GT 标签的训练集数据 (test=False)
    # 因为 shuffle=False 且不打散，所有分块 seg_00, seg_01 将按视频时序完美平铺排列
    dataset = TianmoucStreamingDataset(test=False)
    train_val_data = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False)
    
    # 建立视频导出通道，帧率设为 15 FPS，命名为 train_demo.mp4 以示隔离
    video = cv2.VideoWriter('train_demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 320)) 

    print("🎬 正在启动训练集长视频流双框动态图像渲染测试...")
    
    # 🔒 核心动力学长线记忆指针：记录上一个分块所属的视频主名
    last_video_name = None

    # 设置测试的视频数量上限
    max_videos_to_test = 5  # 最多测试5个视频
    videos_tested = 0
    
    with torch.no_grad():
        for step, data in enumerate(train_val_data):
            cop, td, sd, cop_loc, target_loc = data
            cop, td, sd = cop.cuda(), td.cuda(), sd.cuda()

            # 物理防线：识别当前片段所属的真实长视频序列主名 (例如 "MOT17-02")
            current_seq_id = dataset.seq_ids[step]
            current_video_name = current_seq_id.split('_seg_')[0]

            # 🔒 只有更换全新视频序列时，才执行全盘重置！
            # 处于同一个训练长视频的不同分块内部时，SNN 膜电位和信号质量门控惯性 100% 连续演进！
            if current_video_name != last_video_name:
                # 检查是否已测试足够数量的视频
                if videos_tested >= max_videos_to_test:
                    print(f"✅ 已完成 {max_videos_to_test} 个视频的测试，提前退出。")
                    break
                
                print(f"📡 [场景大阶跃点火] 检测到进入全新训练大场景: {current_video_name}，脉冲状态流归零初始化。")
                print(f"   (当前已测试 {videos_tested}/{max_videos_to_test} 个视频)")
                backbone.reset_stream_state()
                last_video_name = current_video_name
                videos_tested += 1

            base_T_interval = 10
            T_steps = td.shape[-1]
            pred_loc_list = []
            
            # 流式因果向前推理
            for t in range(T_steps):
                is_rgb_available = (t % base_T_interval == 0)
                
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

            # 3. 🟥 🟩 实时双框画面高保真渲染
            for t in range(T_steps):
                plt.gca().clear()
                frame_t = cop_np_seq[..., t]
                if frame_t.max() <= 1.01:
                    frame_t = frame_t * 255.0
                frame_t = np.clip(frame_t, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(frame_t.transpose(1, 2, 0), cv2.COLOR_BGR2RGB)

                for o in range(3):
                    # 🔒 从网络中吐出来和硬盘读出来的都是 [0, 1] 相对值，原地乘以画布物理尺寸还原
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

                    # 🟩 绘制绿色真值框 (只有在有效的目标框下才绘制，过滤全零补齐)
                    if gcx > 0 or gcy > 0:
                        img = cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2) 

                    # 🟥 绘制红色 HNN 脉冲流预测框
                    img = cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2) 

                plt.imshow(img)
                plt.pause(0.01)
                video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    video.release()
    print("📊 成果大片渲染完毕，双框对比成果已安全导出至项目根目录下的 train_demo.mp4！")

if __name__ == "__main__":
    run_val_visual()