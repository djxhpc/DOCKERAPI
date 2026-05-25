import os
import json
import easyocr
import numpy as np
from PIL import Image, ImageOps
from natsort import natsorted

def recognize_benchmark_numbers(image_folder, output_json_path):
    print("正在初始化 EasyOCR 模型...")
    reader = easyocr.Reader(['en'], gpu=True)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    image_files = natsorted(image_files)
    
    if not image_files:
        print(f"錯誤：在 '{image_folder}' 資料夾中找不到支援的圖片檔案。")
        return

    results_dict = {}
    print(f"開始辨識，共 {len(image_files)} 張圖片（已啟用多角度自動校正）...")

    for idx, file_name in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, file_name)
        
        try:
            # 1. 讀取圖片
            with Image.open(image_path) as img:
                # 💡 修正手機拍攝產生的 EXIF 轉向問題
                img = ImageOps.exif_transpose(img)
                img_rgb = img.convert('RGB')
            
            final_number = ""
            best_backup_number = "" # 萬一真的沒 4 位數，留著備用
            
            # 💡 2. 角度暴力破解：嘗試 0度、90度、180度、270度
            # 因為水準點上的數字通常只有一個方向是對的
            rotations = [0, 90, 180, 270]
            
            for angle in rotations:
                # 旋轉圖片 (0度不變, 90度順時針...以此類推)
                if angle == 0:
                    current_img = img_rgb
                else:
                    # 使用 Image.ROTATE_90, 180, 270
                    rot_method = getattr(Image, f"ROTATE_{angle}")
                    current_img = img_rgb.transpose(rot_method)
                
                # 轉成 numpy array 給 EasyOCR
                img_np = np.array(current_img)
                
                # 辨識數字
                ocr_results = reader.readtext(img_np, detail=0, allowlist='0123456789')
                
                # 檢查是否有剛好 4 位數的結果
                four_digit_numbers = [text for text in ocr_results if len(text) == 4]
                
                if four_digit_numbers:
                    final_number = four_digit_numbers[0]
                    # 只要抓到 4 位數，就認定這個角度是對的，跳出旋轉迴圈
                    if angle != 0:
                        print(f"   [提示] 檔案 {file_name} 在旋轉 {angle} 度後成功辨識！")
                    break
                elif ocr_results and not best_backup_number:
                    # 如果沒有 4 位數，先記下第一個看到的數字，當作最後防線
                    best_backup_number = ocr_results[0]
            
            # 如果四個角度都轉完了還是沒有 4 位數的，就拿備用數字
            if not final_number:
                final_number = best_backup_number

            results_dict[file_name] = final_number
            print(f"[{idx}/{len(image_files)}] 檔案: {file_name} => 辨識結果: {final_number}")
            
        except Exception as e:
            results_dict[file_name] = "Error"
            print(f"[{idx}/{len(image_files)}] 檔案: {file_name} => 處理失敗: {str(e)}")

    # 4. 輸出 JSON 
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results_dict, json_file, ensure_ascii=False, indent=4)
        
    print(f"\n辨識完成！結果已成功儲存至: {output_json_path}")

if __name__ == "__main__":
    TARGET_FOLDER = r"D:\Chiayi_AI\一等水準點照片" # 請替換成你的實際中文路徑
    OUTPUT_JSON = r"D:\Chiayi_AI\一等水準點照片\benchmark_results2.json"
    
    if os.path.exists(TARGET_FOLDER):
        recognize_benchmark_numbers(TARGET_FOLDER, OUTPUT_JSON)
    else:
        print(f"找不到路徑: {TARGET_FOLDER}")
