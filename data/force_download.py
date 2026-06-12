import os
import requests
from tqdm import tqdm

files = [
    "README.md",
    "Tianmouc-R_all.zip",
    "Tianmouc-R_test.zip",
    "Tianmouc-R_train.zip",
    "Tianmouc-R_val.zip"
]

repo_url = "https://hf-mirror.com/datasets/ordinarabbit/Tianmouc-R/resolve/main/"
local_dir = "./Tianmouc-R"
os.makedirs(local_dir, exist_ok=True)

for file in files:
    file_url = repo_url + file
    local_path = os.path.join(local_dir, file)
    
    # 检查是否已经下载过且完整（大于1MB视作已存在，避免重复下载你那42GB）
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024*1024:
        print(f"检测到 {file} 已存在，自动跳过...")
        continue
        
    print(f"正在强行下载: {file}")
    response = requests.get(file_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=file
    ) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            f.write(data)
            bar.update(len(data))

print("🎉 所有文件强行下载完毕！")
