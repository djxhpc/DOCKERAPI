import json
import os
import math
from PIL import Image
import matplotlib.pyplot as plt

# ==================== 設定區 ====================
json_file_path = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\111\桿件\桿件埋深-照片\111桿件埋深-照片test桿件埋深-照片_output\yolo_results.json"
# ===============================================

def preview_none_detected_images(json_path):
    # 讀取 JSON 檔案
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到 JSON 檔案 {json_path}")
        return

    results_list = data.get("results_list", [])
    
    # 篩選出所有 status 為 "None Detected" 且檔案路徑確實存在的影像
    valid_images = []
    for item in results_list:
        if item.get("status") == "None Detected":
            abs_path = item.get("absolute_path")
            filename = item.get("filename")
            if abs_path and os.path.exists(abs_path):
                valid_images.append((filename, abs_path))
            else:
                print(f"[警告] 找不到實體檔案，跳過：{abs_path}")

    total_images = len(valid_images)
    print(f"找到 {total_images} 張狀態為 'None Detected' 的影像。開始準備預覽...")

    if total_images == 0:
        print("沒有可顯示的影像。")
        return

    # 設定每頁顯示的網格大小（例如 3 欄 x 3 列 = 9 張圖）
    cols = 3
    rows = 3
    imgs_per_page = cols * rows
    total_pages = math.ceil(total_images / imgs_per_page)

    for page in range(total_pages):
        start_idx = page * imgs_per_page
        end_idx = min(start_idx + imgs_per_page, total_images)
        page_images = valid_images[start_idx:end_idx]

        # 建立 matplotlib 畫布
        fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
        fig.suptitle(f"None Detected 影像預覽 (第 {page+1}/{total_pages} 頁)", fontsize=16)
        
        # 將 2D 的 axes 矩陣拉平，方便跑迴圈
        axes_flat = axes.flatten()

        for i, ax in enumerate(axes_flat):
            if i < len(page_images):
                filename, abs_path = page_images[i]
                try:
                    # 讀取影像並顯示
                    img = Image.open(abs_path)
                    ax.imshow(img)
                    ax.set_title(filename, fontsize=8, wrap=True)
                except Exception as e:
                    ax.text(0.5, 0.5, f"讀取失敗\n{e}", ha='center', va='center')
            else:
                # 如果該頁不滿 9 張，隱藏多餘的格子
                ax.axis('off')
            
            # 隱藏座標軸刻度
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        print(f"正在顯示第 {page+1}/{total_pages} 頁，【請關閉影像視窗】以觀看下一頁...")
        plt.show()  # 這會暫停程式，直到你手動把視窗關掉

    print("所有影像預覽完畢！")

if __name__ == "__main__":
    preview_none_detected_images(json_file_path)
