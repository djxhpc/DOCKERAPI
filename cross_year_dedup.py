"""
cross_year_dedup.py — 跨年代影像去重 + 圖片比較瀏覽器

功能：
  1. 遞迴掃描指定根目錄下所有年代資料夾的圖片
  2. 計算 SHA-256，找出跨年代重複的影像
  3. 輸出 JSON（指定格式）
  4. 提供 Tkinter UI 瀏覽器，可逐筆對比重複圖與原圖

用法：
  python cross_year_dedup.py          # 先掃描去重，再開啟 UI
  python cross_year_dedup.py --ui     # 直接開啟 UI（需已有 JSON）
  python cross_year_dedup.py --scan   # 只掃描，不開 UI
"""

import os
import sys
import json
import hashlib
import argparse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageOps

# =============================================================
# 設定區
# =============================================================
BASE_DIR = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi"
OUTPUT_JSON = os.path.join(BASE_DIR, "111cross_year_duplicates.json")

# 要掃描的年代資料夾名稱（可自行增減）
# YEAR_FOLDERS = ["110", "111", "112", "113", "114", "115"]
YEAR_FOLDERS = ["110"]

# 圖片副檔名
IMG_EXT = ('.jpg', '.jpeg', '.jpe', '.png', '.bmp', '.webp', '.tiff')

# UI 預覽圖片最大尺寸
PREVIEW_MAX_W = 600
PREVIEW_MAX_H = 450


# =============================================================
# 第一部分：掃描與去重
# =============================================================

def get_file_sha256(filepath, chunk_size=8192):
    """計算檔案的 SHA-256"""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return None


def scan_and_dedup(base_dir, year_folders, output_json):
    """
    遞迴掃描 base_dir 下所有 year_folders 的圖片，
    計算 SHA-256，找出重複影像並輸出 JSON。

    JSON 格式：
    {
        "重複檔名.jpg": {
            "sha256": "...",
            "duplicate_of": "原始檔名.jpg"
        },
        ...
    }
    """
    print(f"{'='*60}")
    print(f"  跨年代影像去重")
    print(f"  根目錄: {base_dir}")
    print(f"  年代: {', '.join(year_folders)}")
    print(f"{'='*60}\n")

    if not os.path.exists(base_dir):
        print(f"[錯誤] 根目錄不存在: {base_dir}")
        return {}

    # ── 收集所有圖片路徑 ──────────────────────────────
    all_images = []  # [(relative_display_name, absolute_path), ...]
    scanned_dirs = 0

    for year in year_folders:
        year_dir = os.path.join(base_dir, year)
        if not os.path.isdir(year_dir):
            print(f"  [跳過] 找不到年代資料夾: {year_dir}")
            continue

        for root, dirs, files in os.walk(year_dir):
            # 跳過 _output 子目錄
            dirs[:] = [d for d in dirs if not d.endswith("_output")]
            scanned_dirs += 1
            for f in sorted(files):
                if f.lower().endswith(IMG_EXT):
                    abs_path = os.path.join(root, f)
                    # 顯示用的相對路徑（從 year/ 開始）
                    rel_from_year = os.path.relpath(abs_path, year_dir)
                    display_name = os.path.join(year, rel_from_year)
                    all_images.append((display_name, abs_path))

    total = len(all_images)
    print(f"  掃描資料夾數: {scanned_dirs}")
    print(f"  找到圖片總數: {total}")

    if total == 0:
        print("  沒有找到任何圖片，結束。")
        return {}

    # ── 逐一計算 SHA-256 並比對 ──────────────────────
    # hash_to_first: sha256 → (display_name, abs_path)  （第一個出現的作為「原始」）
    hash_to_first = {}
    duplicates = {}  # 最終輸出格式
    errors = 0

    print(f"\n  開始計算 SHA-256 ...")
    for idx, (display_name, abs_path) in enumerate(all_images, 1):
        sha256 = get_file_sha256(abs_path)
        if sha256 is None:
            errors += 1
            print(f"    [讀取錯誤] {display_name}")
            continue

        if sha256 in hash_to_first:
            # 重複！
            orig_display, orig_abs = hash_to_first[sha256]
            duplicates[display_name] = {
                "sha256": sha256,
                "duplicate_of": orig_display,
                "_abs_path": abs_path,           # UI 用（不寫入 JSON）
                "_orig_abs_path": orig_abs,      # UI 用（不寫入 JSON）
            }
        else:
            hash_to_first[sha256] = (display_name, abs_path)

        if idx % 500 == 0 or idx == total:
            print(f"    進度: {idx}/{total}  (已找到重複: {len(duplicates)})")

    # ── 統計摘要 ──────────────────────────────────────
    unique = len(hash_to_first)
    dup_count = len(duplicates)

    print(f"\n{'─'*40}")
    print(f"  不重複: {unique}")
    print(f"  重複:   {dup_count}")
    print(f"  錯誤:   {errors}")
    print(f"  唯一 SHA-256: {len(hash_to_first)}")

    # ── 統計跨年代重複分布 ──────────────────────────
    if duplicates:
        cross_year_count = 0
        for dup_name, info in duplicates.items():
            dup_year = dup_name.split(os.sep)[0]
            orig_year = info["duplicate_of"].split(os.sep)[0]
            if dup_year != orig_year:
                cross_year_count += 1
        print(f"  其中跨年代重複: {cross_year_count}")

    # ── 寫入 JSON（不含 _abs_path 等內部欄位）────────
    json_output = {}
    for name, info in duplicates.items():
        json_output[name] = {
            "sha256": info["sha256"],
            "duplicate_of": info["duplicate_of"],
        }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)
    print(f"\n  結果已儲存: {output_json}")
    print(f"{'='*60}\n")

    return duplicates


# =============================================================
# 第二部分：Tkinter 圖片比較瀏覽器
# =============================================================

class DuplicateViewer:
    """跨年代重複圖片瀏覽器 — 左右對比原始圖與重複圖"""

    def __init__(self, duplicates_with_paths, output_json):
        self.duplicates = duplicates_with_paths  # dict with _abs_path
        self.output_json = output_json

        # 排序：依年代分組，方便瀏覽
        self.sorted_keys = sorted(
            self.duplicates.keys(),
            key=lambda k: (k.split(os.sep)[0], k)
        )
        self.current_idx = 0
        self.total = len(self.sorted_keys)

        self._build_ui()
        self.filtered_keys = list(self.sorted_keys)
        self.total = len(self.filtered_keys)

        self._load_current()

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("跨年代重複圖片瀏覽器")
        self.root.geometry("1350x800")
        self.root.configure(bg="#2b2b2b")
        self.root.minsize(1100, 650)

        # ── 頂部資訊列 ─────────────────────────────
        top_frame = tk.Frame(self.root, bg="#3c3f41", padx=10, pady=8)
        top_frame.pack(fill=tk.X)

        self.lbl_title = tk.Label(
            top_frame, text="跨年代重複圖片瀏覽器",
            font=("Microsoft JhengHei", 14, "bold"),
            fg="#cccccc", bg="#3c3f41"
        )
        self.lbl_title.pack(side=tk.LEFT)

        self.lbl_count = tk.Label(
            top_frame, text="",
            font=("Consolas", 12), fg="#6cb6ff", bg="#3c3f41"
        )
        self.lbl_count.pack(side=tk.RIGHT)

        # ── SHA-256 資訊列 ─────────────────────────
        info_frame = tk.Frame(self.root, bg="#313335", padx=10, pady=5)
        info_frame.pack(fill=tk.X)
        self.lbl_info = tk.Label(
            info_frame, text="", font=("Consolas", 10),
            fg="#aaaaaa", bg="#313335", anchor="w", justify="left"
        )
        self.lbl_info.pack(fill=tk.X)

        # ── 圖片比較區（左=原始，右=重複）────────
        img_frame = tk.Frame(self.root, bg="#2b2b2b")
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左側：原始圖
        left_frame = tk.LabelFrame(
            img_frame, text="  原始圖片 (Original)  ",
            font=("Microsoft JhengHei", 11, "bold"),
            fg="#8bc34a", bg="#2b2b2b", labelanchor="n"
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.lbl_orig_path = tk.Label(
            left_frame, text="", font=("Consolas", 9),
            fg="#a0a0a0", bg="#2b2b2b", wraplength=550, justify="center"
        )
        self.lbl_orig_path.pack(pady=(5, 2))

        self.canvas_orig = tk.Label(left_frame, bg="#1e1e1e")
        self.canvas_orig.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 右側：重複圖
        right_frame = tk.LabelFrame(
            img_frame, text="  重複圖片 (Duplicate)  ",
            font=("Microsoft JhengHei", 11, "bold"),
            fg="#f44336", bg="#2b2b2b", labelanchor="n"
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.lbl_dup_path = tk.Label(
            right_frame, text="", font=("Consolas", 9),
            fg="#a0a0a0", bg="#2b2b2b", wraplength=550, justify="center"
        )
        self.lbl_dup_path.pack(pady=(5, 2))

        self.canvas_dup = tk.Label(right_frame, bg="#1e1e1e")
        self.canvas_dup.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ── 底部按鈕列 ─────────────────────────────
        btn_frame = tk.Frame(self.root, bg="#3c3f41", padx=10, pady=8)
        btn_frame.pack(fill=tk.X)

        btn_style = {
            "font": ("Microsoft JhengHei", 11),
            "width": 10, "relief": "flat", "cursor": "hand2"
        }

        self.btn_prev = tk.Button(
            btn_frame, text="◀ 上一筆", bg="#555", fg="white",
            command=self._prev, **btn_style
        )
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(
            btn_frame, text="下一筆 ▶", bg="#555", fg="white",
            command=self._next, **btn_style
        )
        self.btn_next.pack(side=tk.LEFT, padx=5)

        self.btn_first = tk.Button(
            btn_frame, text="⏮ 首筆", bg="#444", fg="#ccc",
            command=self._first, **btn_style
        )
        self.btn_first.pack(side=tk.LEFT, padx=5)

        self.btn_last = tk.Button(
            btn_frame, text="末筆 ⏭", bg="#444", fg="#ccc",
            command=self._last, **btn_style
        )
        self.btn_last.pack(side=tk.LEFT, padx=5)

        # 跳轉
        tk.Label(btn_frame, text="跳轉到第", bg="#3c3f41",
                 fg="#ccc", font=("Microsoft JhengHei", 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.entry_jump = tk.Entry(btn_frame, width=6, font=("Consolas", 11))
        self.entry_jump.pack(side=tk.LEFT)
        self.entry_jump.bind("<Return>", lambda e: self._jump())
        tk.Label(btn_frame, text="筆", bg="#3c3f41",
                 fg="#ccc", font=("Microsoft JhengHei", 10)).pack(side=tk.LEFT, padx=(2, 10))
        tk.Button(
            btn_frame, text="前往", bg="#4a7dff", fg="white",
            command=self._jump, font=("Microsoft JhengHei", 10), width=5,
            relief="flat", cursor="hand2"
        ).pack(side=tk.LEFT, padx=5)

        # 篩選
        tk.Label(btn_frame, text="  |  篩選年代:", bg="#3c3f41",
                 fg="#ccc", font=("Microsoft JhengHei", 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.var_filter = tk.StringVar(value="全部")
        years = ["全部"] + sorted(set(
            k.split(os.sep)[0] for k in self.sorted_keys
        ))
        self.combo_filter = ttk.Combobox(
            btn_frame, textvariable=self.var_filter,
            values=years, width=6, state="readonly",
            font=("Consolas", 10)
        )
        self.combo_filter.pack(side=tk.LEFT)
        self.combo_filter.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

        # 匯出按鈕
        tk.Button(
            btn_frame, text="📁 匯出列表", bg="#607d8b", fg="white",
            command=self._export_list, font=("Microsoft JhengHei", 10),
            width=10, relief="flat", cursor="hand2"
        ).pack(side=tk.RIGHT, padx=5)

        # 快捷鍵
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("<Home>", lambda e: self._first())
        self.root.bind("<End>", lambda e: self._last())

    def _resize_image(self, img_path, max_w, max_h):
        """載入圖片並等比例縮放"""
        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
            w, h = img.size
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            # 錯誤時顯示佔位圖
            img = Image.new("RGB", (max_w, max_h), color=(60, 60, 60))
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            draw.text((20, max_h // 2), f"無法載入:\n{os.path.basename(img_path)}\n{e}",
                      fill=(255, 100, 100))
            return ImageTk.PhotoImage(img)

    def _load_current(self):
        """載入並顯示當前筆數的圖片"""
        if self.total == 0:
            return

        key = self.filtered_keys[self.current_idx]
        info = self.duplicates[key]

        # 更新計數
        self.lbl_count.config(
            text=f"第 {self.current_idx + 1} / {self.total} 筆"
        )

        # 更新 SHA-256 資訊
        sha = info["sha256"]
        info_text = (
            f"SHA-256: {sha}\n"
            f"原始檔: {info['duplicate_of']}\n"
            f"重複檔: {key}"
        )
        self.lbl_info.config(text=info_text)

        # 載入原始圖
        orig_path = info.get("_orig_abs_path", "")
        if orig_path and os.path.exists(orig_path):
            self._photo_orig = self._resize_image(orig_path, PREVIEW_MAX_W, PREVIEW_MAX_H)
            self.canvas_orig.config(image=self._photo_orig, text="")
            self.lbl_orig_path.config(text=info["duplicate_of"])
        else:
            self.canvas_orig.config(image="", text="找不到原始圖片", fg="#f44336")
            self.lbl_orig_path.config(text=info["duplicate_of"])

        # 載入重複圖
        dup_path = info.get("_abs_path", "")
        if dup_path and os.path.exists(dup_path):
            self._photo_dup = self._resize_image(dup_path, PREVIEW_MAX_W, PREVIEW_MAX_H)
            self.canvas_dup.config(image=self._photo_dup, text="")
            self.lbl_dup_path.config(text=key)
        else:
            self.canvas_dup.config(image="", text="找不到重複圖片", fg="#f44336")
            self.lbl_dup_path.config(text=key)

    def _prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_current()

    def _next(self):
        if self.current_idx < self.total - 1:
            self.current_idx += 1
            self._load_current()

    def _first(self):
        self.current_idx = 0
        self._load_current()

    def _last(self):
        self.current_idx = self.total - 1
        self._load_current()

    def _jump(self):
        try:
            num = int(self.entry_jump.get())
            if 1 <= num <= self.total:
                self.current_idx = num - 1
                self._load_current()
            else:
                messagebox.showwarning("提示", f"請輸入 1 ~ {self.total} 的數字")
        except ValueError:
            messagebox.showwarning("提示", "請輸入有效的數字")

    def _apply_filter(self):
        """依選擇的年代篩選"""
        selected = self.var_filter.get()
        if selected == "全部":
            self.filtered_keys = list(self.sorted_keys)
        else:
            self.filtered_keys = [
                k for k in self.sorted_keys
                if k.split(os.sep)[0] == selected
            ]
        self.total = len(self.filtered_keys)
        self.current_idx = 0
        if self.total > 0:
            self._load_current()
        else:
            self.lbl_count.config(text="無符合條件的項目")
            self.canvas_orig.config(image="", text="")
            self.canvas_dup.config(image="", text="")
            self.lbl_info.config(text="")

    def _export_list(self):
        """匯出目前篩選結果的文字列表"""
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt"), ("CSV", "*.csv")],
            initialfile="duplicate_list.txt"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("重複檔名\tSHA-256\t原始檔名\n")
            for key in self.filtered_keys:
                info = self.duplicates[key]
                f.write(f"{key}\t{info['sha256']}\t{info['duplicate_of']}\n")
        messagebox.showinfo("匯出完成", f"已匯出 {self.total} 筆到:\n{path}")

    def run(self):
        self.filtered_keys = list(self.sorted_keys)
        self.total = len(self.filtered_keys)
        if self.total > 0:
            self._load_current()
        self.root.mainloop()


# =============================================================
# 主程式入口
# =============================================================

def load_duplicates_with_paths(json_path, base_dir):
    """
    從 JSON 載入重複資料，並嘗試推導絕對路徑供 UI 使用。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    duplicates = {}
    for name, info in data.items():
        abs_path = os.path.join(base_dir, name)
        orig_name = info["duplicate_of"]
        orig_abs = os.path.join(base_dir, orig_name)

        duplicates[name] = {
            "sha256": info["sha256"],
            "duplicate_of": orig_name,
            "_abs_path": abs_path,
            "_orig_abs_path": orig_abs,
        }
    return duplicates


def main():
    parser = argparse.ArgumentParser(description="跨年代影像去重工具")
    parser.add_argument("--scan", action="store_true", help="只執行掃描去重")
    parser.add_argument("--ui", action="store_true", help="直接開啟 UI（需已有 JSON）")
    parser.add_argument("--base", type=str, default=BASE_DIR, help="根目錄路徑")
    parser.add_argument("--output", type=str, default=OUTPUT_JSON, help="輸出 JSON 路徑")
    args = parser.parse_args()

    # ✅ 不用 global，直接用區域變數接住參數
    base = args.base
    output = args.output

    if args.ui:
        # 直接開 UI
        if not os.path.exists(output):
            print(f"[錯誤] 找不到 JSON: {output}")
            print("請先執行掃描: python cross_year_dedup.py --scan")
            sys.exit(1)
        print(f"載入 {output} ...")
        dup = load_duplicates_with_paths(output, base)  # ← 多傳 base
        if not dup:
            print("JSON 中沒有重複資料。")
            sys.exit(0)
        print(f"載入 {len(dup)} 筆重複記錄，開啟瀏覽器...")
        viewer = DuplicateViewer(dup, output)
        viewer.run()

    elif args.scan:
        # 只掃描
        scan_and_dedup(base, YEAR_FOLDERS, output)

    else:
        # 預設：掃描 + 開 UI
        dup = scan_and_dedup(base, YEAR_FOLDERS, output)
        if not dup:
            # 嘗試從已有 JSON 載入
            if os.path.exists(output):
                dup = load_duplicates_with_paths(output, base)  # ← 多傳 base
        if dup:
            print("開啟圖片比較瀏覽器...\n")
            viewer = DuplicateViewer(dup, output)
            viewer.run()
        else:
            print("沒有找到重複圖片，結束。")


if __name__ == "__main__":
    main()
