from pathlib import Path

# 要整理的圖片資料夾路徑
folder_path = Path(r"D:\Chiayi_AI\117\尺規有問題")

# 輸出的 txt 檔案路徑
output_txt = Path(r"D:\Chiayi_AI\117\尺規有問題\image_list.txt")

# 支援的圖片副檔名
image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

# 取得圖片檔名並依名稱排序
image_names = sorted(
    [file.name for file in folder_path.iterdir()
     if file.is_file() and file.suffix.lower() in image_extensions]
)

# 寫入 txt 檔案
with output_txt.open("w", encoding="utf-8") as f:
    for name in image_names:
        f.write(name + "\n")

print(f"完成！共寫入 {len(image_names)} 個圖片名稱到 {output_txt}")
