import os
import json
from PIL import Image
import imagehash

# ==================== 設定區 ====================
# 設定主要的工作根目錄
base_dir = r"D:\Chiayi_AI"
# 要處理的年份資料夾名稱
years = ["110", "111", "113", "114", "115"]    
# years = ["112-0"]
# ===============================================

def process_folder_images(folder_path, folder_name, year_prefix):
    """
    處理單一資料夾內的照片比對與 JSON 產出（加入毀損檔案紀錄）
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    
    # 找出該資料夾下「直接包含」的所有照片
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(folder_path, f))]
    
    if not files:
        return  # 如果這層資料夾沒有照片，就跳過不處理

    print(f"\n正在處理: {folder_path}")
    
    seen_hashes = {}
    remain_files = []
    dup_files = []
    corrupted_files = []  # 用來記錄損毀的檔案

    # 排序檔案名稱
    files.sort()

    # 使用 pHash 進行比對
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            with Image.open(file_path) as img:
                img_hash = imagehash.phash(img)
            
            if img_hash in seen_hashes:
                dup_files.append(file)
            else:
                seen_hashes[img_hash] = file_path
                remain_files.append(file)
        except Exception as e:
            print(f"  [錯誤] 無法處理檔案 {file}，原因: {e}")
            # 捕獲異常，將檔名丟進損毀清單
            corrupted_files.append(file)

    # 建立輸出資料夾名稱
    output_dir_name = f"{year_prefix}{folder_name}_output"
    output_folder_path = os.path.join(folder_path, output_dir_name)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 1. 產出 output.json (未重複照片)
    remain_json_data = [{"images": f} for f in remain_files]
    remain_output_path = os.path.join(output_folder_path, "output.json")
    with open(remain_output_path, 'w', encoding='utf-8') as jf:
        json.dump(remain_json_data, jf, ensure_ascii=False, indent=4)

    # 2. 產出 repeat_images.json (重複照片)
    dup_json_data = [{"images": f} for f in dup_files]
    dup_output_path = os.path.join(output_folder_path, "repeat_images.json")
    with open(dup_output_path, 'w', encoding='utf-8') as jf:
        json.dump(dup_json_data, jf, ensure_ascii=False, indent=4)

    # 3. 產出 corrupted_images.json (損毀照片) -> 只有在有毀損檔案時才建立
    if corrupted_files:
        corrupted_json_data = [{"images": f} for f in corrupted_files]
        corrupted_output_path = os.path.join(output_folder_path, "corrupted_images.json")
        with open(corrupted_output_path, 'w', encoding='utf-8') as jf:
            json.dump(corrupted_json_data, jf, ensure_ascii=False, indent=4)
        print(f"  -> [注意] 已產出損毀照片報表：{corrupted_output_path} (共 {len(corrupted_files)} 張)")

    print(f"  -> 成功產生報表於: {output_folder_path}")
    print(f"     (未重複: {len(remain_files)} 張, 重複: {len(dup_files)} 張, 損毀: {len(corrupted_files)} 張)")


def main():
    for year in years:
        year_dir = os.path.join(base_dir, year)
        if not os.path.exists(year_dir):
            print(f"找不到年份資料夾: {year_dir}，跳過。")
            continue
            
        print(f"\n==================== 開始掃描 {year} 年度 ====================")
        
        for root, dirs, files in os.walk(year_dir):
            # 排除掉我們自己產生的 _output 資料夾
            dirs[:] = [d for d in dirs if not d.endswith("_output")]
            
            folder_name = os.path.basename(root)
            if folder_name == year:
                folder_name = f"{year}年度"
                
            process_folder_images(root, folder_name, year)

    print("\n============== 所有年度影像掃描與報表產出作業完成 ==============")


if __name__ == "__main__":
    main()
