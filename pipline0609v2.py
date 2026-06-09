"""
pipeline_0525.py — 影像去重與多模態辨識整合管道

工作流程（每個資料夾獨立執行）:
  Step 1  去重     : pHash 比對去重，產生 output.json / repeat_images.json / corrupted_images.json
  Step 2a 尺規     : YOLO 辨識直/橫尺是否交叉，產生 yolo_results.json
  Step 2b 座標OCR  : Tesseract OCR 辨識 N/E/H 座標，產生 coord_ocr_results.json
  Step 2c 水準點   : EasyOCR 辨識 4 位數水準點號碼，產生 benchmark_results.json

增量設計:
  - 去重  : 新照片才計算 hash，已知照片沿用 hashes.json 的紀錄
  - 辨識  : 只對 output.json 中尚未有結果的照片處理，結果持續附加至各 JSON
"""
import os
import json
import re
import hashlib
import unicodedata
import numpy as np
from PIL import Image, ImageOps
import imagehash
import cv2


# =============================================
# 設定區 — 請依實際情況修改
# =============================================
# ── 處理任務清單 ─────────────────────────────────────────────
# 每個 dict 代表一個獨立的處理任務：
#   source_dir : 要遞迴處理的根目錄（直接指定絕對路徑）
#   steps      : 要執行的步驟清單，可含 "coord_ocr" / "benchmark" / "ruler_ac" / "yolo"
#   output_dir : 全域整合 JSON 的輸出目錄
#   job_name   : 輸出檔名前綴與日誌顯示名稱
JOBS = [
    # {
    #     "source_dir": r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\111\桿件\桿件埋深-照片",
    #     "steps":      ["ruler_ac", "yolo"],
    #     "output_dir": r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\117\新輸出",
    #     "job_name":   "111桿件埋深-照片test",
    # },
    # 新增更多任務範例：
    {
        "source_dir": r"D:\work2\0526ocr\testorror\尺規",
        "steps":      ["ruler_ac", "yolo"],
        "output_dir": r"D:\work2\0526ocr\testorror\新輸出",
        "job_name":   "117尺規",
    },
]

# ── 工具路徑 ──────────────────────────────────
YOLO_MODEL_PATH    = r"D:\work2\0526ocr\best.pt"
RULER_YOLO_ENABLED = True  # False = 只跑 ruler_ac（A+C法），跳過 YOLO 尺規辨識

# ── YOLO垂直尺 + 接觸偵測（ruler_yc）─────────────────
# 只用 YOLO 偵測垂直尺（class 1），再在尺的周圍搜尋是否有物體靠著，
# 完全不需要判斷水平物體是什麼種類。
RULER_YC_ENABLED      = True
RULER_YC_MODEL_PATH   = YOLO_MODEL_PATH  # 共用同一模型
RULER_YC_CONF         = 0.1    # 垂直尺偵測信心閾值
RULER_YC_EXPAND_RATIO = 2.5    # 搜尋帶向外延伸倍數（相對於尺的寬度）
RULER_YC_LINE_RATIO   = 0.10   # 搜尋帶內最小有效水平線長度（佔帶寬比例）

# ── 水準點 YOLO 輔助裁切（訓練完模型後將 BENCHMARK_YOLO_CROP 改為 True）──
BENCHMARK_YOLO_CROP       = False                      # False = 停用；True = 啟用
BENCHMARK_YOLO_MODEL_PATH = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\117\尺規best.pt" # 訓練好的一等水準點偵測模型
BENCHMARK_YOLO_CONF       = 0.25   # 偵測信心閾值（低一點避免漏偵測）
BENCHMARK_YOLO_PADDING    = 0.3    # 裁切框四周額外留邊比例（相對於框的寬/高）

# ── A+C 尺規辨識參數 ───────────────────────────
# 垂直黃色尺的 HSV 範圍（可用 cv2 取樣後微調）
YELLOW_HSV_LOWER = [18, 100, 100]
YELLOW_HSV_UPPER = [38, 255, 255]
# 水平橫條最小長度（佔圖寬比例），過短的線段視為背景雜訊
HOUGH_MIN_LINE_RATIO = 0.15

# ── 步驟路由 ──────────────────────────────────
# 資料夾名稱（或路徑）包含 key 時，執行 value 中的步驟。
# 步驟代號: "yolo" | "ruler_ac" | "coord_ocr" | "benchmark"
# 多個關鍵字可同時命中，步驟取聯集。
FOLDER_RULES = {
    # "埋深":  ["ruler_ac", "yolo"],
    # "桿件埋深-照片":  ["ruler_ac", "yolo"],
    "尺規":  ["ruler_ac", "yolo"],
    # "一等水準點照片": ["benchmark"],
    # "測量讀數照片": ["coord_ocr"],
    # "測量坐標讀數": ["coord_ocr"],
    # "測量拍照-測量照片讀數照片": ["coord_ocr"],
    # "測量照片讀數-照片": ["coord_ocr"],
    # "坐標測量-照片": ["coord_ocr"],
    # "測量坐標讀數": ["coord_ocr"],

    # "測量照片讀數-照片2": ["coord_ocr"],
    #本周先執行(0527~0529)
    # "測試": ["coord_ocr"],
    # "驗證測量讀數照片": ["coord_ocr"],

}
# 無關鍵字命中時的預設步驟
DEFAULT_STEPS = []

# 【沒命中關鍵字時的「去重開關」】
# "OPEN"  : 開啟去重（沒命中關鍵字時，雖然不做辨識，但依然幫這個資料夾執行 Step 1 影像去重）
# "CLOSE" : 關閉去重（沒命中關鍵字時，連去重都不做，直接跳過、忽略這個資料夾）
NO_MATCH_DEDUPLICATE_SWITCH = "CLOSE"

# ── 影像分類模型路由（訓練完模型後將 USE_CLASSIFY_MODEL 改為 True）──────
USE_CLASSIFY_MODEL   = False                      # False = 使用 FOLDER_RULES；True = 使用分類模型
CLASSIFY_MODEL_PATH  = r"D:\Chiayi_AI\YOLO_dataset\影像分類best.pt" # classify_train.py 訓練輸出的 best.pt
CLASSIFY_CONF_THRESH = 0    # 低於此信心值 → 自動 fallback 到 FOLDER_RULES 0.6
# 分類類別名稱 → 步驟（key 必須與 classify_train.py 的 CLASS_DIRS key 一致）
CLASSIFY_CLASS_STEPS = {
    "埋深":   ["ruler_ac", "yolo"],
    # "水準點": ["benchmark"],
    # "座標":   ["coord_ocr"],
    # "其他":   [],
}


# ── 疑似重複偵測（對抗縮放/對比變造）──────────────────
PHASH_SUSPECTED_ENABLED   = True  # False = 停用
PHASH_SUSPECTED_THRESHOLD = 8     # pHash(16) Hamming 距離 ≤ 此值視為疑似重複（建議 5–10）

# ── 座標 OCR YOLO 格式分類（訓練完模型後將 COORD_YOLO_ENABLED 改為 True）──
COORD_YOLO_ENABLED    = False                        # False = 停用；True = 啟用
COORD_YOLO_MODEL_PATH = r"D:\work2\0526ocr\best2.pt"
# COORD_YOLO_MODEL_PATH = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\117\判斷格式分類best2.pt" # 訓練好的座標格式分類模型
COORD_YOLO_CONF       = 0.1  # 低於此信心值 → fallback 到全格式模式（類別 9）
# 模型輸出類別名稱 → 格式代號（1–8），依訓練時設定的 CLASS_DIRS key 填寫；未列出的 → 9（全模式）
COORD_YOLO_CLASS_MAP  = {
    "local_NEH":     1,   # 本地N / 本地E / 本地H
    "simple_NEZ":    2,   # 純 N E Z
    "multi_NEZ":     3,   # N E 高程 / N E Z / 縱橫高程
    "local_surface": 4,   # 本地N/E 高程 / 地表N/E/H ////已解決+未
    "north_east":    5,   # 北座標 東座標 高（繁簡）/ 北東高度 / 北東高 ////已解決
    "bottom_left":   6,   # 抓左下角 N E H  ////已解決
    "mixed_NEZ":     7,   # N E Z 或 北東高程 ////已解決
    "nez_only":      8,   # N E Z ////已解決
}
# =============================================

_IMG_EXT = ('.jpg', '.jpeg', '.jpe', '.png', '.bmp', '.webp', '.tiff')



# ─────────────────────────────────────────────────────────────
# STEP 1  去重 (remain0525json 邏輯 + 增量支援)
# ─────────────────────────────────────────────────────────────

def _load_json_set(path):
    """讀取 [{"images": "..."}, ...] 格式 JSON，回傳 set of filenames"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return {item["images"] for item in json.load(f) if item.get("images")}
    return set()


def _save_json_list(path, filenames):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([{"images": n} for n in sorted(filenames)], f, ensure_ascii=False, indent=4)


def _get_file_sha256(filepath, chunk_size=8192):
    """計算檔案的 SHA-256 雜湊值（純二進位讀取，極快）"""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception:
        return None

def dedup_folder(folder_path, folder_name, job_name):
    """
    對 folder_path 內的照片進行純 SHA-256 精確去重（檔案 100% 相同才算重複）。
    已知照片從 hashes.json 讀取，不重複計算；新照片才計算並比對。
    回傳 output.json 路徑；資料夾無照片則回傳 None。
    """
    # 假設 _IMG_EXT 在外部已定義，例如: _IMG_EXT = ('.jpg', '.jpeg', '.png')
    all_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(_IMG_EXT) and os.path.isfile(os.path.join(folder_path, f))
    )
    if not all_files:
        return None

    out_dir = os.path.join(folder_path, f"{job_name}{folder_name}_output")
    os.makedirs(out_dir, exist_ok=True)

    remain_path  = os.path.join(out_dir, "output.json")
    dup_path     = os.path.join(out_dir, "repeat_images.json")
    corrupt_path = os.path.join(out_dir, "corrupted_images.json")
    hashes_path  = os.path.join(out_dir, "hashes.json")

    # 讀取已處理的歷史狀態
    known_remain  = _load_json_set(remain_path)
    known_corrupt = _load_json_set(corrupt_path)
    
    # 讀取重複圖片歷史紀錄 (相容舊格式)
    # known_dup_dict = {}
    # if os.path.exists(dup_path):
    #     with open(dup_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #         if isinstance(data, list):
    #             known_dup_dict = {k: {"sha256": "未知", "duplicate_of": "未知"} for k in data}
    #         elif isinstance(data, dict):
    #             known_dup_dict = data
    # 找到這一段並修改：
    known_dup_dict = {}  # 預設空值，避免 dup_path 不存在時 UnboundLocalError
    if os.path.exists(dup_path):
        with open(dup_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                # 修正處：確保 key 是字串，而不是整個字典
                known_dup_dict = {}
                for item in data:
                    # 假設列表結構是 [{"images": "xxx"}, ...]
                    if isinstance(item, dict) and "images" in item:
                        known_dup_dict[item["images"]] = {"sha256": "未知", "duplicate_of": "未知"}
                    # 如果列表結構只是單純的字串列表
                    elif isinstance(item, str):
                        known_dup_dict[item] = {"sha256": "未知", "duplicate_of": "未知"}
            elif isinstance(data, dict):
                known_dup_dict = data
    known_dup_keys = set(known_dup_dict.keys())
    known_all = known_remain | known_dup_keys | known_corrupt

    # 讀取已知的歷史 Hashes
    saved_hashes = {}
    if os.path.exists(hashes_path):
        with open(hashes_path, 'r', encoding='utf-8') as f:
            saved_hashes = json.load(f)

    new_files = [f for f in all_files if f not in known_all]
    if not new_files:
        print(f"  [{folder_path}] 無新照片，跳過去重。")
        return remain_path if os.path.exists(remain_path) else None

    print(f"\n正在進行 SHA-256 精確去重: {folder_path}（新檔案 {len(new_files)} 個）")

    # 重建已存在的不重複檔案 SHA-256 索引表 (相容舊版相容格式)
    existing_sha256 = {}
    for fn in known_remain:
        if fn in saved_hashes:
            val = saved_hashes[fn]
            # 如果舊資料是字典且包含 sha256
            if isinstance(val, dict) and "sha256" in val:
                existing_sha256[val["sha256"]] = fn
            # 如果舊資料直接是字串（如果是舊版 pHash，這段會失效，新跑的檔案會重新計算 SHA 後覆蓋）
            elif isinstance(val, str) and len(val) == 64: 
                existing_sha256[val] = fn

    new_remain, new_corrupt = set(), set()
    new_dup_dict = {}

    for fn in new_files:
        fp = os.path.join(folder_path, fn)
        
        file_sha256 = _get_file_sha256(fp)
        if file_sha256 is None:
            print(f"  [錯誤] {fn}: 無法讀取檔案")
            new_corrupt.add(fn)
            continue

        # 核心比對：只認 SHA-256 
        if file_sha256 in existing_sha256:
            orig_fn = existing_sha256[file_sha256]
            new_dup_dict[fn] = {
                "sha256": file_sha256,
                "duplicate_of": orig_fn
            }
        else:
            # 這是全新的 SHA-256，立刻加入索引，防範同批次後面有重複檔案
            existing_sha256[file_sha256] = fn
            saved_hashes[fn] = {
                "sha256": file_sha256
            }
            new_remain.add(fn)

    # 合併新舊結果
    final_remain  = known_remain  | new_remain
    final_corrupt = known_corrupt | new_corrupt
    
    final_dup_dict = known_dup_dict.copy()
    final_dup_dict.update(new_dup_dict)

    # 寫入結果 JSON 檔案
    _save_json_list(remain_path, final_remain)
    if final_corrupt:
        _save_json_list(corrupt_path, final_corrupt)

    with open(dup_path, 'w', encoding='utf-8') as f:
        json.dump(final_dup_dict, f, ensure_ascii=False, indent=4)

    with open(hashes_path, 'w', encoding='utf-8') as f:
        json.dump(saved_hashes, f, ensure_ascii=False, indent=4)

    print(f"  -> 不重複: {len(final_remain)} | 重複: {len(final_dup_dict)} | 損毀: {len(final_corrupt)}")

    # ── Stage 2: pHash(16) 疑似重複偵測 ──────────────────────────────
    if PHASH_SUSPECTED_ENABLED:
        suspected_path = os.path.join(out_dir, "suspected_duplicates.json")
        known_suspected = {}
        if os.path.exists(suspected_path):
            with open(suspected_path, 'r', encoding='utf-8') as f:
                known_suspected = json.load(f)

        # 重建所有「目前不重複照片」的 pHash16 索引
        # 格式: { phash_str: first_filename }
        phash_index = {}

        def _get_phash16(filepath):
            try:
                return str(imagehash.phash(Image.open(filepath), hash_size=16))
            except Exception:
                return None

        # 先載入已知 pHash（從 hashes.json 的 phash16 欄位）
        for fn in final_remain:
            ph = saved_hashes.get(fn, {}).get("phash16") if isinstance(saved_hashes.get(fn), dict) else None
            if ph is None:
                fp = os.path.join(folder_path, fn)
                ph = _get_phash16(fp)
                if ph and isinstance(saved_hashes.get(fn), dict):
                    saved_hashes[fn]["phash16"] = ph
            if ph:
                phash_index[ph] = fn

        new_suspected = {}
        # 對新照片進行疑似比對（只掃 new_remain，避免重複掃舊圖）
        for fn in new_remain:
            if fn in known_suspected:
                continue
            fp = os.path.join(folder_path, fn)
            ph_str = saved_hashes.get(fn, {}).get("phash16") if isinstance(saved_hashes.get(fn), dict) else None
            if ph_str is None:
                ph_str = _get_phash16(fp)
                if ph_str and isinstance(saved_hashes.get(fn), dict):
                    saved_hashes[fn]["phash16"] = ph_str
            if ph_str is None:
                continue
            ph_obj = imagehash.hex_to_hash(ph_str)
            best_dist, best_match = PHASH_SUSPECTED_THRESHOLD + 1, None
            for existing_ph_str, existing_fn in phash_index.items():
                if existing_fn == fn:
                    continue
                dist = ph_obj - imagehash.hex_to_hash(existing_ph_str)
                if dist <= PHASH_SUSPECTED_THRESHOLD and dist < best_dist:
                    best_dist, best_match = dist, existing_fn
            if best_match:
                new_suspected[fn] = {
                    "phash16": ph_str,
                    "similar_to": best_match,
                    "distance": int(best_dist)
                }

        if new_suspected:
            final_suspected = {**known_suspected, **new_suspected}
            with open(suspected_path, 'w', encoding='utf-8') as f:
                json.dump(final_suspected, f, ensure_ascii=False, indent=4)
            print(f"  -> 疑似重複（pHash16）: {len(final_suspected)} 筆（新增 {len(new_suspected)}）→ {suspected_path}")
        elif not known_suspected:
            pass  # 無疑似重複且無舊紀錄，不建立空檔案
        else:
            print(f"  -> 疑似重複（pHash16）: {len(known_suspected)} 筆（無新增）")

        # 更新 hashes.json（補入 phash16 欄位）
        with open(hashes_path, 'w', encoding='utf-8') as f:
            json.dump(saved_hashes, f, ensure_ascii=False, indent=4)

    return remain_path


# ─────────────────────────────────────────────────────────────
# STEP 2a  YOLO 尺規辨識 (findrulerv2 邏輯)
# ─────────────────────────────────────────────────────────────

def _bbox_crossed(bv, bh):
    vx1, vy1, vx2, vy2 = bv
    hx1, hy1, hx2, hy2 = bh
    # 嚴格重疊
    if min(vx2, hx2) > max(vx1, hx1) and min(vy2, hy2) > max(vy1, hy1):
        return True
    vcx = (vx1 + vx2) / 2
    hcy = (hy1 + hy2) / 2
    # 中心點落在對方範圍內
    if (vy1 <= hcy <= vy2) and (hx1 <= vcx <= hx2):
        return True
    # 50px 寬鬆容差
    if (vy1 - 50 <= hcy <= vy2 + 50) and (hx1 - 50 <= vcx <= hx2 + 50):
        return True
    return False


def run_yolo(output_json_path, folder_path, model_path):
    """
    增量 YOLO 尺規辨識。
    只處理 output.json 中尚未有結果的照片。
    結果附加至 <output_dir>/yolo_results.json。
    """
    from ultralytics import YOLO as YOLOModel

    out_dir      = os.path.dirname(output_json_path)
    results_path = os.path.join(out_dir, "yolo_results.json")

    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    else:
        report = {
            "summary": {k: 0 for k in [
                "total_images", "both_detected_crossed",
                "vertical_only", "horizontal_only", "none_detected", "file_read_errors"
            ]},
            "results_list": []
        }

    done = {r["filename"] for r in report["results_list"]}

    with open(output_json_path, 'r', encoding='utf-8') as f:
        new_items = [it for it in json.load(f) if it.get("images") and it["images"] not in done]

    if not new_items:
        print("  [YOLO] 無新照片需處理。")
        return

    print(f"  [YOLO] 載入模型: {model_path}")
    model = YOLOModel(model_path)
    print(f"  [YOLO] 開始辨識 {len(new_items)} 張新照片...")

    sm = report["summary"]
    for idx, item in enumerate(new_items, 1):
        fn  = item["images"]
        fp  = os.path.join(folder_path, fn)
        rec = {"filename": fn, "absolute_path": os.path.abspath(fp)}

        try:
            pil = Image.open(fp).convert("RGB")
            arr = np.array(pil)
        except Exception as e:
            rec.update(status="Error", reason=str(e))
            sm["file_read_errors"] += 1
            report["results_list"].append(rec)
            continue

        preds = model.predict(source=arr, verbose=False)[0]
        vb, hb = [], []
        for box in (preds.boxes or []):
            if float(box.conf[0].item()) > 0.05:
                xyxy = box.xyxy[0].tolist()
                (vb if int(box.cls[0].item()) == 1 else hb).append(xyxy)

        vc, hc = len(vb), len(hb)
        if vc == 0 and hc == 0:
            status = "None Detected";           sm["none_detected"] += 1
        elif vc > 0 and hc == 0:
            status = "Vertical Only";           sm["vertical_only"] += 1
        elif vc == 0 and hc > 0:
            status = "Horizontal Only";         sm["horizontal_only"] += 1
        elif any(_bbox_crossed(b1, b2) for b1 in vb for b2 in hb):
            status = "Both Detected (Crossed)"; sm["both_detected_crossed"] += 1
        else:
            # 可視範圍不重疊 → 視同 None Detected
            status = "None Detected";           sm["none_detected"] += 1

        rec.update(status=status, detections={"vertical_count": vc, "horizontal_count": hc})
        report["results_list"].append(rec)

        if idx % 100 == 0 or idx == len(new_items):
            print(f"  [YOLO] 進度: {idx}/{len(new_items)}")

    sm["total_images"] = len(report["results_list"])
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"  [YOLO] 完成！結果儲存至: {results_path}")


# ─────────────────────────────────────────────────────────────
# STEP 2a-alt  A+C 尺規辨識（無需訓練模型）
#   C: HSV 黃色遮罩 → 垂直黃色直尺
#   A: Hough 直線 + 長度篩選 → 水平灰色橫條
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# STEP 2a-alt  A+C 尺規辨識（無需訓練模型）
#   C: HSV 黃色遮罩 → 垂直/水平黃色直尺（方向不限）
#   A: Hough 直線群組分析 → 確認為尺規（非單根棍子）
# ─────────────────────────────────────────────────────────────

def _group_parallel_lines(lines, axis='h', gap=20):
    """
    將接近同位置的平行線群組化。
    axis='h': 依 y 座標分群（水平線）
    axis='v': 依 x 座標分群（垂直線）
    回傳 list of groups，每個 group 是 list of lines。
    """
    if not lines:
        return []

    key_fn = (lambda l: (l[1] + l[3]) / 2) if axis == 'h' else (lambda l: (l[0] + l[2]) / 2)
    sorted_lines = sorted(lines, key=key_fn)

    groups = []
    current = [sorted_lines[0]]
    for line in sorted_lines[1:]:
        if abs(key_fn(line) - key_fn(current[-1])) <= gap:
            current.append(line)
        else:
            groups.append(current)
            current = [line]
    groups.append(current)
    return groups


def _ruler_band_has_yellow(img_hsv, y_lo, y_hi, x_lo, x_hi,
                            yellow_lower, yellow_upper, ratio=0.03):
    """
    檢查指定矩形區域內是否有足夠比例的黃色像素。
    """
    roi = img_hsv[max(0, y_lo):y_hi, max(0, x_lo):x_hi]
    if roi.size == 0:
        return False
    mask = cv2.inRange(roi,
                       np.array(yellow_lower, dtype=np.uint8),
                       np.array(yellow_upper, dtype=np.uint8))
    return (mask > 0).mean() >= ratio


def _has_ruler_stripes(img_gray, y_lo, y_hi, x_lo, x_hi, min_alternations=4):
    """
    檢查指定水平帶內是否有尺規的等間距刻度條紋。
    沿 x 方向計算每列灰階均值，形成 1D 信號後統計明暗交替次數。
    尺規有規律黑白條紋（≥ min_alternations 次交替），棍子/桿件無此特徵。
    """
    roi = img_gray[max(0, y_lo):y_hi, max(0, x_lo):x_hi]
    if roi.size == 0 or roi.shape[1] < 10:
        return False

    # 每列均值 → 1D 亮度信號
    col_means = roi.mean(axis=0).astype(float)

    # 平滑（避免感光雜訊造成假交替）
    kernel_size = max(3, len(col_means) // 40)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(col_means, np.ones(kernel_size) / kernel_size, mode='valid')
    if len(smoothed) < 4:
        return False

    # 以中位數為分界，計算高低交替次數
    threshold = np.median(smoothed)
    above = smoothed > threshold
    alternations = int(np.sum(np.diff(above.astype(int)) != 0))
    return alternations >= min_alternations


def _detect_contact_near_vertical(gray, vbox, h_img, w_img,
                                   expand_ratio=2.5, min_line_ratio=0.10):
    """
    在已知垂直尺的 bbox 左右延伸帶內，偵測是否有任何物體靠著尺。

    判斷邏輯：
    1. 線段必須水平（±20°）
    2. 至少有一端延伸到尺規 x 範圍外（排除尺自身邊緣誤觸發）
    3. 排除「貫穿型背景物件」：若線段同時在尺規左右兩側都延伸超過
       PASSTHROUGH_RATIO × 尺寬，代表棍子/管子是斜穿整個畫面的背景物，
       與尺在不同深度，視為不接觸。
    """
    PASSTHROUGH_RATIO = 1.5   # 雙側超出此倍數的尺寬 → 視為背景貫穿物

    vx1, vy1, vx2, vy2 = [int(v) for v in vbox]
    ruler_w = max(vx2 - vx1, 1)
    ext  = int(ruler_w * expand_ratio)
    sx1  = max(0, vx1 - ext)
    sx2  = min(w_img, vx2 + ext)
    sy1  = max(0, vy1)
    sy2  = min(h_img, vy2)

    zone  = gray[sy1:sy2, sx1:sx2]
    if zone.size == 0:
        return False

    edges = cv2.Canny(zone, 50, 150, apertureSize=3)
    min_len = max(10, int((sx2 - sx1) * min_line_ratio))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=40,
                             minLineLength=min_len,
                             maxLineGap=20)
    if lines is None:
        return False

    passthrough_thresh = ruler_w * PASSTHROUGH_RATIO

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if not (angle < 20 or angle > 160):
            continue  # 非水平線

        abs_x1 = sx1 + min(x1, x2)
        abs_x2 = sx1 + max(x1, x2)

        left_overhang  = vx1 - abs_x1   # 正值 = 延伸到尺規左側
        right_overhang = abs_x2 - vx2   # 正值 = 延伸到尺規右側

        # 條件 2：至少一端超出尺規範圍
        if not (left_overhang > 3 or right_overhang > 3):
            continue

        # 條件 3：排除雙側都大幅超出的貫穿型背景物件
        if left_overhang > passthrough_thresh and right_overhang > passthrough_thresh:
            continue

        return True

    return False


def _detect_rulers_ac(img_path):
    """
    偵測單張圖片中的直尺（C法：黃色HSV）與水平尺規（A法：Hough線群）。
    
    改進重點：
    1. C法同時支援垂直與水平方向的黃色尺規（圖片方向不固定）
    2. A法要求水平線必須是「多條平行線的群組」或「帶有黃色紋理」，
       排除單根棍子/桿子
    
    回傳 (has_vertical, has_horizontal, is_crossed, v_boxes, h_ruler_bands)
    """
    

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False, False, False, [], []

    # EXIF 轉正（手機直拍圖 EXIF 方向未套用會導致垂直/水平誤判）
    try:
        pil_tmp = Image.open(img_path)
        pil_tmp = ImageOps.exif_transpose(pil_tmp)
        img = cv2.cvtColor(np.array(pil_tmp.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ══════════════════════════════════════════════════════
    # C法：HSV 黃色遮罩 → 偵測黃色尺規（垂直或水平皆可）
    # ══════════════════════════════════════════════════════
    mask = cv2.inRange(hsv,
                       np.array(YELLOW_HSV_LOWER, dtype=np.uint8),
                       np.array(YELLOW_HSV_UPPER, dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    v_boxes   = []  # 垂直方向黃色尺規
    h_y_boxes = []  # 水平方向黃色尺規（來自C法）

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 400:
            continue
        rect  = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        # 確保 rh 永遠是長邊
        if rw > rh:
            rw, rh = rh, rw
        aspect = rh / max(rw, 1)

        x, y, bw, bh = cv2.boundingRect(c)

        if aspect > 2.5 and rh > h_img * 0.12:
            # 長邊顯著 → 判斷朝向
            angle = rect[2]  # OpenCV minAreaRect 角度 (-90, 0]
            # 判斷是否接近垂直（角度在 -90°~-70° 或 -20°~0° 之間視為垂直）
            # minAreaRect 回傳的角度定義：短邊與水平軸夾角
            # 若矩形接近垂直，angle 會接近 -90° 或接近 0° 但長邊為垂直
            if bh >= bw:  # bounding box 高度 >= 寬度 → 視為垂直尺
                v_boxes.append((x, y, x + bw, y + bh))
            else:         # bounding box 寬度 > 高度 → 視為水平尺
                h_y_boxes.append((x, y, x + bw, y + bh))

    has_vertical   = len(v_boxes) > 0
    # 黃色水平尺（C法直接確認）
    has_h_yellow   = len(h_y_boxes) > 0

    # ══════════════════════════════════════════════════════
    # A法：Hough 直線 → 嚴格篩選水平尺規（排除棍子）
    # ══════════════════════════════════════════════════════
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_len = int(w_img * HOUGH_MIN_LINE_RATIO)
    lines   = cv2.HoughLinesP(edges, 1, np.pi / 180,
                               threshold=60,
                               minLineLength=min_len,
                               maxLineGap=25)

    raw_h_lines = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 20 or angle > 160:   # ±20° 視為水平
                raw_h_lines.append((x1, y1, x2, y2))

    # ── 關鍵過濾：排除孤立單線（棍子特徵）──────────────────
    #
    # 策略一：群組化 — 要求同一 Y 帶內至少有 N 條平行線
    #         （尺規有刻度，會產生多條平行邊緣線）
    # 策略二：黃色紋理驗證 — 線條附近有黃色像素
    # 兩者任一成立即視為有效水平尺規
    # ─────────────────────────────────────────────────────

    MIN_PARALLEL_LINES = 2   # 至少 2 條平行線才視為尺規
    BAND_HALF          = 15  # 線條上下各 15px 的帶狀區域做黃色驗證

    valid_h_ruler_bands = []

    groups = _group_parallel_lines(raw_h_lines, axis='h', gap=20)
    for group in groups:
        if len(group) >= MIN_PARALLEL_LINES:
            # 群組內線條足夠多 → 確定是尺規
            valid_h_ruler_bands.append(group)
            continue

        # 單條線 → 黃色紋理 AND 刻度條紋同時滿足才視為尺規（排除棍子/桿件）
        x1, y1, x2, y2 = group[0]
        y_center = int((y1 + y2) / 2)
        x_lo     = min(x1, x2)
        x_hi     = max(x1, x2)
        has_yellow   = _ruler_band_has_yellow(hsv,
                                              y_center - BAND_HALF,
                                              y_center + BAND_HALF,
                                              x_lo, x_hi,
                                              YELLOW_HSV_LOWER, YELLOW_HSV_UPPER,
                                              ratio=0.03)
        has_stripes  = _has_ruler_stripes(gray,
                                          y_center - BAND_HALF * 2,
                                          y_center + BAND_HALF * 2,
                                          x_lo, x_hi,
                                          min_alternations=4)
        if has_yellow and has_stripes:
            valid_h_ruler_bands.append(group)
        # 否則視為棍子/桿子，丟棄

    has_h_hough = len(valid_h_ruler_bands) > 0

    # 水平尺：C法黃色 OR A法嚴格Hough
    has_horizontal = has_h_yellow or has_h_hough

    # 合併所有水平線段（供交叉判斷）
    all_h_lines = []
    for g in valid_h_ruler_bands:
        all_h_lines.extend(g)
    for bx1, by1, bx2, by2 in h_y_boxes:
        # 將水平黃色 bbox 轉成中心線
        mid_y = (by1 + by2) // 2
        all_h_lines.append((bx1, mid_y, bx2, mid_y))

    # ══════════════════════════════════════════════════════
    # 交叉判斷（直立尺 ∩ 水平尺）
    # ══════════════════════════════════════════════════════
    is_crossed = False
    CROSS_TOL  = 40  # 交叉判斷容差（px），避免尺/線稍微錯位就漏判
    if has_vertical and has_horizontal:
        for hx1, hy1, hx2, hy2 in all_h_lines:
            hcy   = (hy1 + hy2) / 2
            hx_lo = min(hx1, hx2)
            hx_hi = max(hx1, hx2)
            for vx1, vy1, vx2, vy2 in v_boxes:
                vcx = (vx1 + vx2) / 2
                if ((vy1 - CROSS_TOL <= hcy <= vy2 + CROSS_TOL) and
                        (hx_lo - CROSS_TOL <= vcx <= hx_hi + CROSS_TOL)):
                    is_crossed = True
                    break
            if is_crossed:
                break

    return has_vertical, has_horizontal, is_crossed, v_boxes, all_h_lines


def run_ruler_ac(output_json_path, folder_path):
    """
    增量 A+C 尺規辨識。
    只處理 output.json 中尚未有結果的照片。
    結果附加至 <output_dir>/ruler_ac_results.json。
    """
    out_dir      = os.path.dirname(output_json_path)
    results_path = os.path.join(out_dir, "ruler_ac_results.json")

    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    else:
        report = {
            "summary": {k: 0 for k in [
                "total_images", "both_detected_crossed",
                "vertical_only", "horizontal_only", "none_detected", "file_read_errors"
            ]},
            "results_list": []
        }

    done = {r["filename"] for r in report["results_list"]}

    with open(output_json_path, 'r', encoding='utf-8') as f:
        new_items = [it for it in json.load(f) if it.get("images") and it["images"] not in done]

    if not new_items:
        print("  [尺規A+C] 無新照片需處理。")
        return

    print(f"  [尺規A+C] 開始辨識 {len(new_items)} 張新照片...")

    sm = report["summary"]
    for idx, item in enumerate(new_items, 1):
        fn  = item["images"]
        fp  = os.path.join(folder_path, fn)
        rec = {"filename": fn, "absolute_path": os.path.abspath(fp)}

        try:
            has_v, has_h, crossed, _, _ = _detect_rulers_ac(fp)
        except Exception as e:
            rec.update(status="Error", reason=str(e))
            sm["file_read_errors"] += 1
            report["results_list"].append(rec)
            continue

        if not has_v and not has_h:
            status = "None Detected";           sm["none_detected"] += 1
        elif has_v and not has_h:
            status = "Vertical Only";           sm["vertical_only"] += 1
        elif not has_v and has_h:
            status = "Horizontal Only";         sm["horizontal_only"] += 1
        elif crossed:
            status = "Both Detected (Crossed)"; sm["both_detected_crossed"] += 1
        else:
            # 可視範圍不重疊 → 視同 None Detected
            status = "None Detected";           sm["none_detected"] += 1

        rec.update(status=status, detections={"has_vertical": has_v, "has_horizontal": has_h})
        report["results_list"].append(rec)

        if idx % 100 == 0 or idx == len(new_items):
            print(f"  [尺規A+C] 進度: {idx}/{len(new_items)}")

    sm["total_images"] = len(report["results_list"])
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"  [尺規A+C] 完成！結果儲存至: {results_path}")


# ─────────────────────────────────────────────────────────────
# STEP 2a-yc  YOLO垂直尺 + 接觸偵測（ruler_yc）
#   只偵測垂直尺（class 1），在尺的鄰近帶用 Hough 判斷是否有物體靠著，
#   不需要分類水平物件種類。
# ─────────────────────────────────────────────────────────────

def run_ruler_yolo_contact(output_json_path, folder_path, model_path):
    """
    增量 YOLO垂直尺＋接觸偵測。
    結果附加至 <output_dir>/ruler_yc_results.json。
    """
    from ultralytics import YOLO as YOLOModel

    out_dir      = os.path.dirname(output_json_path)
    results_path = os.path.join(out_dir, "ruler_yc_results.json")

    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    else:
        report = {
            "summary": {k: 0 for k in [
                "total_images", "both_detected_crossed",
                "vertical_only", "none_detected", "file_read_errors"
            ]},
            "results_list": []
        }

    done = {r["filename"] for r in report["results_list"]}

    with open(output_json_path, 'r', encoding='utf-8') as f:
        new_items = [it for it in json.load(f) if it.get("images") and it["images"] not in done]

    if not new_items:
        print("  [尺規YC] 無新照片需處理。")
        return

    print(f"  [尺規YC] 載入模型: {model_path}")
    model = YOLOModel(model_path)
    print(f"  [尺規YC] 開始辨識 {len(new_items)} 張新照片...")

    sm = report["summary"]
    for idx, item in enumerate(new_items, 1):
        fn  = item["images"]
        fp  = os.path.join(folder_path, fn)
        rec = {"filename": fn, "absolute_path": os.path.abspath(fp)}

        try:
            # EXIF 轉正
            pil = Image.open(fp)
            pil = ImageOps.exif_transpose(pil).convert("RGB")
            arr = np.array(pil)
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            h_img, w_img = gray.shape[:2]
        except Exception as e:
            rec.update(status="Error", reason=str(e))
            sm["file_read_errors"] += 1
            report["results_list"].append(rec)
            continue

        preds = model.predict(source=arr, verbose=False)[0]
        vboxes = [
            box.xyxy[0].tolist()
            for box in (preds.boxes or [])
            if float(box.conf[0].item()) >= RULER_YC_CONF
            and int(box.cls[0].item()) == 1   # class 1 = 垂直尺
        ]

        if not vboxes:
            status = "None Detected"; sm["none_detected"] += 1
        elif any(_detect_contact_near_vertical(
                     gray, vb, h_img, w_img,
                     RULER_YC_EXPAND_RATIO, RULER_YC_LINE_RATIO)
                 for vb in vboxes):
            status = "Both Detected (Crossed)"; sm["both_detected_crossed"] += 1
        else:
            status = "Vertical Only"; sm["vertical_only"] += 1

        rec.update(status=status, detections={"vertical_count": len(vboxes)})
        report["results_list"].append(rec)

        if idx % 100 == 0 or idx == len(new_items):
            print(f"  [尺規YC] 進度: {idx}/{len(new_items)}")

    sm["total_images"] = len(report["results_list"])
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"  [尺規YC] 完成！結果儲存至: {results_path}")


# ─────────────────────────────────────────────────────────────
# STEP 2b  座標 OCR (orgocr 邏輯)
# ─────────────────────────────────────────────────────────────

_NUM = r'(-?\d+\.\d+|-?\d+)'

# 依優先順序排列的座標 pattern，每組為 (N_regex, E_regex, H_regex)。
# 具體標籤（北座標、地表N）排前，避免通用字母（N/E/H）誤匹配。
_COORD_PATTERNS = [
    (r'北座標[:=\s]*'      + _NUM, r'東座標[:=\s]*'      + _NUM, r'高程[:=\s]*'         + _NUM),
    (r'縱軸[:=\s]*'        + _NUM, r'橫軸[:=\s]*'        + _NUM, r'高程[:=\s]*'         + _NUM),
    (r'地表.*?[NN][:=\s]*' + _NUM, r'地表.*?[EE][:=\s]*' + _NUM, r'地表.*?[HH][:=\s]*' + _NUM),
    (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'本地.*?[HH][:=\s]*' + _NUM),
    (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'高程[:=\s]*'         + _NUM),
    (r'北[:=\s]*'          + _NUM, r'東[:=\s]*'          + _NUM, r'高程?[:=\s]*'        + _NUM),
    (r'北[:=\s]*'          + _NUM, r'東[:=\s]*'          + _NUM, r'高度[:=\s]*'         + _NUM),
    (r'[NN][:=\s]*'        + _NUM, r'[EE][:=\s]*'        + _NUM, r'[ZZ][:=\s]*'         + _NUM),
    (r'[NN][:=\s]*'        + _NUM, r'[EE][:=\s]*'        + _NUM, r'[HH][:=\s]*'         + _NUM),
]

# 各格式代號對應的 pattern 組（YOLO 格式分類後選用）
# 格式 6（左下角裁切）在 run_coord_ocr 中特殊處理，此處不列
_CLASS_PATTERNS = {
1: [  # 本地 N/E/H
        # 說明：利用 .*? 讓中間可以夾雜任意文字（包含第一個H），進而讓最後的 [HH] 匹配到位置最下方的 H
        (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'本地.*?.[HH][:=\s]*' + _NUM),
        (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'高程[:=\s]*'         + _NUM),
    ],
    2: [  # 純 N E Z
        (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'[ZZ][:=\s]*' + _NUM),
    ],
    3: [  # N E 高程 / N E Z / 縱軸 橫軸 高程
        (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'高程[:=\s]*' + _NUM),
        (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'[ZZ][:=\s]*' + _NUM),
        (r'縱軸[:=\s]*' + _NUM, r'橫軸[:=\s]*' + _NUM, r'高程[:=\s]*' + _NUM),
        # (r'縱軸[:=\s]*' + _NUM, r'[橫横][軸由][:=\s]*' + _NUM, r'高程[:=\s]*' + _NUM),
    ],
    4: [  # 本地N/E 高程 / 地表N/E/H / 坐标北值/东值/H值
        (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'高程[:=\s]*'         + _NUM),
        (r'地表.*?[NN][:=\s]*' + _NUM, r'地表.*?[EE][:=\s]*' + _NUM, r'地表.*?[HH][:=\s]*' + _NUM),
    ],
    5: [  # 北座標/東座標/高（繁簡）/ 北東高度 / 北東高 / N E h（小寫h儀器格式）
        (r'北[座坐][標标][:=\s]*' + _NUM, r'[東东][座坐][標标][:=\s]*' + _NUM, r'高[座坐][標标][:=\s]*' + _NUM),
        (r'北[座坐][標标][:=\s]*' + _NUM, r'[東东][座坐][標标][:=\s]*' + _NUM, r'高程[:=\s]*'           + _NUM),
        (r'北[:=\s]*'             + _NUM, r'東[:=\s]*'                 + _NUM, r'高度[:=\s]*'           + _NUM),
        (r'北[:=\s]*'             + _NUM, r'東[:=\s]*'                 + _NUM, r'高[:=\s]*'             + _NUM),
        (r'[Nn][:=\s]*'           + _NUM, r'[Ee][:=\s]*'              + _NUM, r'[Hh][:=\s]+'           + _NUM),
    ],
    # 7: [  # N E Z / 北 東 高程
    #     (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'[ZZ][:=\s]*' + _NUM),
    #     # (r'北[:=\s]*'   + _NUM, r'東[:=\s]*'   + _NUM, r'高程[:=\s]*' + _NUM),
    # ],
    7: [  # mixed_NEZ 特殊客製：E(6位)在上，N(7位)在中間，Z在後
        # 第一優先：有明確標籤，且文字順序為 E 在 N 前面
        (r'[EE][:=\s\n]*' + _NUM + r'.*?[NN][:=\s\n]*' + _NUM + r'.*?[ZZ27z][:=\s\n]*' + _NUM), 
        # 第二優先：標準跨行規則（但順序改為 E -> N -> Z）
        (r'[EE][:=\s\n]*' + _NUM, r'[NN][:=\s\n]*' + _NUM, r'[ZZ27z][:=\s\n]*' + _NUM)
    ],
    8: [  # N E Z
        (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'[ZZ][:=\s]*' + _NUM),
    ],
}




def _ocr_with_paddle(image_path, engine):
    """RapidOCR 辨識，image_path 可為路徑字串或 numpy array，回傳合併文字（信心值 > 0.3）"""
    result, _ = engine(image_path)
    if not result:
        return ""
    return "\n".join(item[1] for item in result if float(item[2]) > 0.3)


def _match_coord(text, patterns):
    """從 OCR 文字依 patterns 依序提取 N/E/H_Z，回傳最佳結果 dict。"""
    
    best = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}
    m = re.search(r'NEZ[:=\s]*' + _NUM + r',' + _NUM + r',' + _NUM, text.replace(' ', ''))
    if m:
        return {"N": m.group(1), "E": m.group(2), "H_Z": m.group(3)}
    for np_, ep_, hp_ in patterns:
        mn = re.search(np_, text, re.IGNORECASE)
        me = re.search(ep_, text, re.IGNORECASE)
        mh = re.search(hp_, text, re.IGNORECASE)
        if mn and me and mh:
            return {"N": mn.group(1), "E": me.group(1), "H_Z": mh.group(1)}
        if mn and best["N"]   == "N/A": best["N"]   = mn.group(1)
        if me and best["E"]   == "N/A": best["E"]   = me.group(1)
        if mh and best["H_Z"] == "N/A": best["H_Z"] = mh.group(1)
    return best


def _strip_dms(text):
    """
    移除 OCR 文字中的度分秒（DMS）格式座標，避免緯經度數值被誤認為 TWD97 座標。
    例如: "23°27'14.0603°" → 刪除，以免 OCR 把它合併成 "232714.0603" 當作東座標。
    """
    dms_pattern = re.compile(
        r'\d{1,3}\s*[°\*oO]\s*\d{1,2}\s*[\'′]\s*\d{1,2}(?:\.\d+)?[""′\']?\s*[NSEWnsew]?',
        re.IGNORECASE
    )
    return dms_pattern.sub('', text)


def _validate_twd97(res):
    """
    回傳 True 表示 N/E/H_Z 均符合 TWD97 格式：
      - N=7位整數，且數值在 2,400,000–2,800,000（台灣含離島）
      - E=6位整數，且數值在 147,000–350,000
      - H_Z 含小數點
    """
    def _id(v): return len(v.split('.')[0].lstrip('-'))
    if not (
        res["N"]   != "N/A" and _id(res["N"])   == 7 and
        res["E"]   != "N/A" and _id(res["E"])   == 6 and
        res["H_Z"] != "N/A" and '.' in res["H_Z"]
    ):
        return False
    try:
        n_val = int(res["N"].split('.')[0])
        e_val = int(res["E"].split('.')[0])
        return 2_400_000 <= n_val <= 2_800_000 and 147_000 <= e_val <= 350_000
    except (ValueError, AttributeError):
        return False


def _digit_fallback_twd97(text):
    """
    從 OCR 文字依 TWD97 位數特徵提取座標：
      N = 第一個整數部分恰好 7 位的數字
      E = N 之後，優先選「緊接下一個數字就是合法 H（1~4位整數、含小數）」的 6 位數；
          若所有候選都找不到，退而選第一個 6 位數。
          （此規則可排除緯度DMS殘骸：OCR 把 23°27'14.0603 合併為 232714.0603，
            其後緊接的是另一個 6 位 E 而不是 H，真正的 E 後面緊接才是 H。）
      H_Z = E 之後第一個含小數點且整數部分 1~4 位（且非 0.xxx）的數字
    """
    def _is_valid_h(v):
        if '.' not in v: return False
        ip = v.split('.')[0].lstrip('-')
        return ip != '0' and 1 <= len(ip) <= 4

    all_m = list(re.finditer(r'-?\d+(?:\.\d+)?', text))
    cn = ce = ch = None
    n_idx = e_idx = -1

    # Step 1: 找 N（第一個 7 位整數）
    # Step 1: 找 N（數值介於 2,000,000 到 3,000,000 之間的 7 位整數）
    for i, m in enumerate(all_m):
        v_str = m.group()
        # 移除潛在的小數點部分，取整數部分進行判斷
        clean_v = v_str.split('.')[0].lstrip('-')
        
        # 檢查是否為 7 位數
        if len(clean_v) == 7 and cn is None:
            val = int(clean_v)
            # 檢查數值範圍
            if 2000000 <= val <= 3000000:
                cn = v_str
                n_idx = i
                break

    if cn:
        # Step 2: 收集 N 之後所有 6 位候選
        e_candidates = []
        for i, m in enumerate(all_m):
            if i > n_idx:
                val_str = m.group().split('.')[0].lstrip('-')
                if len(val_str) == 6:
                    val = int(val_str)
                    if 100000 <= val <= 400000:
                        e_candidates.append((i, m.group()))

        # 優先：緊接下一個數字是合法 H 的候選
        for e_i, e_v in e_candidates:
            if e_i + 1 < len(all_m) and _is_valid_h(all_m[e_i + 1].group()):
                ce = e_v; e_idx = e_i
                ch = all_m[e_i + 1].group()
                break

        # 退而：取第一個 6 位候選
        if not ce and e_candidates:
            ce = e_candidates[0][1]; e_idx = e_candidates[0][0]

    # Step 3: 若 H 未找到，往後搜尋
    if ce and not ch:
        for i, m in enumerate(all_m):
            if i <= e_idx: continue
            v = m.group()
            if _is_valid_h(v):
                ch = v; break

    return {
        "N":   cn if cn else "N/A",
        "E":   ce if ce else "N/A",
        "H_Z": ch if ch else "N/A",
    }


def _extract_coord(image_path, engine, patterns=None):
    """OCR 後依 patterns 提取座標（patterns 預設為 _COORD_PATTERNS）。"""
    text = _ocr_with_paddle(image_path, engine)
    return _match_coord(text, patterns if patterns is not None else _COORD_PATTERNS)


# def _extract_coord_bottom_left(image_path, engine, crop_ratio=0.45):
#     """裁切圖片左下角（寬/高各 crop_ratio 比例）後 OCR 提取座標。"""
#     with Image.open(image_path) as img:
#         img = ImageOps.exif_transpose(img).convert('RGB')
#         w, h = img.size
#         crop = img.crop((0, int(h * (1 - crop_ratio)), int(w * crop_ratio), h))
#     text = _ocr_with_paddle(np.array(crop), engine)
#     return _match_coord(text, _COORD_PATTERNS)
def _extract_coord_bottom_left(image_path, engine, crop_ratio=0.45):
    """裁切圖片左下角後 OCR 提取座標。
    為了防止最後一位數字因貼近邊緣而被切掉或忽略，將水平裁切寬度稍微擴大（乘上 1.25 倍緩衝）。
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            w, h = img.size
            
            # 高度維持 0.45 比例（從底部往上算 45%）
            y_start = int(h * (1 - crop_ratio))
            # 關鍵修正：寬度從原本的 w * 0.45 擴大到 w * 0.55，防止右側數值最後一位（如 5）被切到
            x_end = min(int(w * (crop_ratio * 1.25)), w) 
            
            crop = img.crop((0, y_start, x_end, h))
        text = _ocr_with_paddle(np.array(crop), engine)
    except Exception as e:
        text = ""

    # ── [DEBUG] 打印出局部裁切後，RapidOCR 真正看到的文字 ──
    print(f"\n--- [DEBUG Class 6 OCR 文字] 檔案: {os.path.basename(image_path)} ---")
    print(text if text else "(完全沒有辨識出任何文字)")
    print("--------------------------------------------------")

    # 1. 第一層防線：用標準跨行規則（寬度擴大後，英文字母 N/E/H 有機會進來了）
    bottom_left_patterns = [
        (r'N[:=\s\n]*' + _NUM, r'E[:=\s\n]*' + _NUM, r'H[:=\s\n]*' + _NUM),
        (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'[HH][:=\s]*' + _NUM)
    ]
    res = _match_coord(text, bottom_left_patterns)

    # 2. 第二層防線：如果還是有欄位 N/A，改用純數字行阻擊邏輯
    if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        valid_nums = []
        for line in lines:
            # 使用 re.search 提取出行內符合 _NUM 的部分，排除可能一起被切進來的右側文字或雜訊
            # 例如將 "2606210.1205 CAD" 提取出 "2606210.1205"
            m_num = re.search(_NUM, line)
            if m_num:
                valid_nums.append(m_num.group(1))
        
        # 依 Class 6 儀器特性：由上到下依序填入 N, E, H（驗證 N=7位、E=6位、H含小數）
        if len(valid_nums) >= 3:
            v0, v1, v2 = valid_nums[0], valid_nums[1], valid_nums[2]
            def _id(v): return len(v.split('.')[0].lstrip('-'))
            if _id(v0) == 7 and _id(v1) == 6 and '.' in v2:
                res["N"]   = v0
                res["E"]   = v1
                res["H_Z"] = v2

    # 3. 第三層防線：如果連數字都漏抓，啟動全圖 Fallback 掃描
    if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
        print(f"  [提示] 檔案 {os.path.basename(image_path)} 局部辨識失敗，啟動全圖 Fallback 掃描...")
        full_text = _ocr_with_paddle(image_path, engine)
        res = _match_coord(full_text, _COORD_PATTERNS)
        if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
            res = _match_coord(full_text, bottom_left_patterns)

    return res


def _extract_coord_local_neh(image_path, engine):
    """
    Class 1 (local_NEH) 專用。回傳 (res_dict, full_text)，讓呼叫端可重用 full_text。
    找到「本地E」數值的位置後，只在其後方搜尋 H。
    狀態列的 H 格式為 "H: 0.013"（冒號分隔），座標 H 為 "H  9.6263"（空格分隔），
    使用 \\s+ 只匹配空格，排除冒號格式的狀態列 H。
    """
    text = _ocr_with_paddle(image_path, engine)

    print(f"\n--- [DEBUG Class 1 OCR 文字] 檔案: {os.path.basename(image_path)} ---")
    print(text if text else "(完全沒有辨識出任何文字)")
    print("--------------------------------------------------")

    n_match = re.search(r'本地.*?[NN][:=\s]*' + _NUM, text, re.IGNORECASE)
    e_match = re.search(r'本地.*?[EE][:=\s]*' + _NUM, text, re.IGNORECASE)

    res = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}
    if n_match:
        res["N"] = n_match.group(1)
    if e_match:
        res["E"] = e_match.group(1)
        # 只在本地E數值之後搜尋 H，且只接受空格分隔（\s+）
        # 排除狀態列 "H: 0.013"（冒號分隔）這類誤判
        text_after_e = text[e_match.end():]
        # 第一層：有 H 標籤（空格分隔，排除狀態列的 H: 冒號格式）
        h_match = re.search(r'[HH]\s+' + _NUM, text_after_e, re.IGNORECASE)
        if not h_match:
            # 第二層：H 標籤被 OCR 吃掉，直接取第一個出現在行首的數字
            h_match = re.search(r'^\s*' + _NUM, text_after_e, re.MULTILINE)
        if h_match:
            res["H_Z"] = h_match.group(1)

    # H 仍未找到 → 改找高程
    if res["H_Z"] == "N/A":
        h2 = re.search(r'高程[:=\s]*' + _NUM, text, re.IGNORECASE)
        if h2:
            res["H_Z"] = h2.group(1)

    # TWD97 位數驗證：N=7位、E=6位、H_Z含小數；不符合則用位數 fallback
    if not _validate_twd97(res):
        fb = _digit_fallback_twd97(text)
        if fb["N"] != "N/A": res["N"]   = fb["N"]
        if fb["E"] != "N/A": res["E"]   = fb["E"]
        if fb["H_Z"] != "N/A": res["H_Z"] = fb["H_Z"]

    return res, text


def run_coord_ocr(output_json_path, folder_path, engine, coord_yolo):
    """
    增量座標 OCR（RapidOCR 版）。
    engine     : 已初始化的 RapidOCR 實例（由 main() 傳入）
    coord_yolo : 已載入的格式分類 YOLO 模型（由 main() 傳入，None = 停用）
    若 COORD_YOLO_ENABLED，先用分類模型判斷格式代號（1–8），再套用對應 pattern。
      - 格式 6：裁切左下角後辨識。
      - 格式 1–5, 7–8：先嘗試對應 _CLASS_PATTERNS；若三欄未全中則 fallback 到 _COORD_PATTERNS。
      - 格式 9（未知 / 信心不足）：直接用 _COORD_PATTERNS 全模式。
    """
    out_dir      = os.path.dirname(output_json_path)
    results_path = os.path.join(out_dir, "coord_ocr_results.json")

    existing = {}
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)

    with open(output_json_path, 'r', encoding='utf-8') as f:
        new_items = [it for it in json.load(f) if it.get("images") and it["images"] not in existing]

    if not new_items:
        print("  [座標OCR] 無新照片需處理。")
        return

    yolo_device = 'cuda'
    if coord_yolo is not None:
        try:
            coord_yolo.to('cuda')
        except Exception:
            yolo_device = 'cpu'

    print(f"  [座標OCR] 開始辨識 {len(new_items)} 張新照片...")
    for idx, item in enumerate(new_items, 1):
        fn = item["images"]
        fp = os.path.join(folder_path, fn)
        try:
            # ── Step A: YOLO 格式分類 ─────────────────────────────
            # class_id = 9  # 預設：全格式 fallback

            # # 優先：從檔名讀取 class id（例如 photo_c1.jpg → class 1）
            # _fn_cls = re.search(r'_c([1-9])(?=[_.])', fn)
            # if _fn_cls:
            #     class_id = int(_fn_cls.group(1))
            #     print(f"  [座標OCR] 檔名指定類別: class {class_id} ({fn})")
            # elif coord_yolo:
            #     try:
            #         pred     = coord_yolo(fp)[0]
            #         conf     = float(pred.probs.top1conf)
            #         cls_name = pred.names[pred.probs.top1]
            #         if conf >= COORD_YOLO_CONF:
            #             class_id = COORD_YOLO_CLASS_MAP.get(cls_name, 9)
            #     except Exception:
            #         class_id = 9

            # ── 原始版本（未加入檔名判斷）──
            class_id = 9
            if coord_yolo:
                try:
                    pred     = coord_yolo(fp, device=yolo_device)[0]
                    conf     = float(pred.probs.top1conf)
                    cls_name = pred.names[pred.probs.top1]
                    if conf >= COORD_YOLO_CONF:
                        class_id = COORD_YOLO_CLASS_MAP.get(cls_name, 9)
                except Exception:
                    class_id = 9

            # ── Step B: 依格式提取座標 ────────────────────────────
            # ── Step B: 依格式提取座標（加強類別 2, 8 的純 NEZ 辨識） ────────────────────
            if class_id == 6:
                res = _extract_coord_bottom_left(fp, engine)
            elif class_id == 1:
                res, full_text_c1 = _extract_coord_local_neh(fp, engine)
                if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
                    res = _match_coord(full_text_c1, _COORD_PATTERNS)
            else:
                text = _ocr_with_paddle(fp, engine)
                # NFKC 正規化：一次將全形字母/數字/符號（Ａ→A、Ｅ→E、１→1、：→:等）轉為標準形式
                text = unicodedata.normalize('NFKC', text)
                text = _strip_dms(text)  # 移除度分秒格式（緯/經度），防止被誤認為 TWD97 座標

                if class_id in [2]:
                    print(f"\n--- [DEBUG Class 2 OCR 文字] 檔案: {os.path.basename(fp)} ---")
                    print(text if text else "(完全沒有辨識出任何文字)")
                    print("--------------------------------------------------")

                    # 最優先：NEZ: x, y, z 逗號格式（E 在前、N 在後，與 class 7 相同處理）
                    _nez_c2_matched = re.search(
                        r'NEZ\s*[:\s]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)',
                        text, re.IGNORECASE
                    )
                    if _nez_c2_matched:
                        _c2_cands = [_nez_c2_matched.group(1), _nez_c2_matched.group(2), _nez_c2_matched.group(3)]
                        _c2_n = _c2_e = _c2_z = None
                        for c in _c2_cands:
                            il = len(c.split('.')[0].lstrip('-'))
                            if il == 7 and _c2_n is None: _c2_n = c
                            elif il == 6 and _c2_e is None: _c2_e = c
                            elif 1 <= il <= 4 and _c2_z is None: _c2_z = c
                        if _c2_n and _c2_e and _c2_z:
                            res = {"N": _c2_n, "E": _c2_e, "H_Z": _c2_z}
                            print(f"  [Class2 NEZ逗號格式] N={_c2_n}, E={_c2_e}, Z={_c2_z}")
                            _nez_c2_matched = True  # 標記成功，跳過後續解析
                        else:
                            _nez_c2_matched = None  # 位數驗證失敗，繼續往下

                    if not _nez_c2_matched:
                        cleaned_text = text.replace(" ", "")

                        # 第一優先：緊湊格式（N/E/Z 直接相連，無換行）
                        pattern_nez_strict = (
                            r'[NN](?::|=)?' + _NUM +
                            r'[EE](?::|=)?' + _NUM +
                            r'[ZZ27z](?::|=)?' + _NUM
                        )
                        m_strict = re.search(pattern_nez_strict, cleaned_text, re.IGNORECASE)

                        if m_strict:
                            res = {"N": m_strict.group(1), "E": m_strict.group(2), "H_Z": m_strict.group(3)}
                        else:
                            # 第二優先：標籤與數值分行（N 獨立在行首，避免誤抓 σN/σE/σZ）
                            # (?:^|\n)\s* 確保 N/E/Z 出現在行首，σN 前有 σ 字元故不符合
                            n_m = re.search(r'(?:^|\n)\s*[NN]\s*[\n:=]\s*' + _NUM, text, re.IGNORECASE)
                            e_m = re.search(r'(?:^|\n)\s*[EE]\s*[\n:=]\s*' + _NUM, text, re.IGNORECASE)
                            z_m = re.search(r'(?:^|\n)\s*[ZZ27z]\s*[\n:=]\s*' + _NUM, text, re.IGNORECASE)
                            res = {
                                "N":   n_m.group(1) if n_m else "N/A",
                                "E":   e_m.group(1) if e_m else "N/A",
                                "H_Z": z_m.group(1) if z_m else "N/A",
                            }

                    # 第三優先：N/E/Z 標籤被錯開或遺漏，或抓到的值不符合 TWD97 位數格式，改用位數 fallback
                    if not _validate_twd97(res):
                        fb = _digit_fallback_twd97(text)
                        if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                            res = fb
                        
                elif class_id == 3:
                    print(f"\n--- [DEBUG Class 3 OCR 文字] 檔案: {os.path.basename(fp)} ---")
                    print(text if text else "(完全沒有辨識出任何文字)")
                    print("--------------------------------------------------")

                    res = _match_coord(text, _CLASS_PATTERNS[3])
                    if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
                        res = _match_coord(text, _COORD_PATTERNS)

                    # 第三優先：文字殘缺、順序混亂，或抓到的值不符合 TWD97 格式（含 H_Z 無小數），改用位數 fallback
                    if not _validate_twd97(res):
                        fb = _digit_fallback_twd97(text)
                        if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                            res = fb
                elif class_id == 7:
                    # ── Class 7 優先裁切底部資訊欄再 OCR（避免地圖點標籤干擾）──
                    _c7_bottom_text = ""
                    try:
                        with Image.open(fp) as _c7_img:
                            _c7_img = ImageOps.exif_transpose(_c7_img).convert('RGB')
                            _c7_w, _c7_h = _c7_img.size
                            # 底部約 22% 為綠色資訊欄（點號/解算/NEZ）
                            _c7_crop = _c7_img.crop((0, int(_c7_h * 0.78), _c7_w, _c7_h))
                        _c7_bottom_text = _ocr_with_paddle(np.array(_c7_crop), engine)
                    except Exception:
                        pass

                    print(f"\n--- [DEBUG Class 7 底部裁切 OCR] 檔案: {os.path.basename(fp)} ---")
                    print(_c7_bottom_text if _c7_bottom_text else "(底部裁切無文字)")
                    print(f"--- [DEBUG Class 7 全圖 OCR 文字] ---")
                    print(text if text else "(完全沒有辨識出任何文字)")
                    print("--------------------------------------------------")

                    # 底部裁切版若有 NEZ 關鍵字，優先用裁切版文字；否則用全圖文字
                    _c7_text = _c7_bottom_text if re.search(r'NEZ', _c7_bottom_text, re.IGNORECASE) else text

                    res = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}

                    # ============================================================
                    # 【最優先】NEZ: x, y, z 逗號分隔格式（如 "NEZ: 207130.674, 2595082.704, 297.498"）
                    # ============================================================
                    nez_comma_match = re.search(
                        r'NEZ\s*[:\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)',
                        _c7_text, re.IGNORECASE
                    )
                    if nez_comma_match:
                        raw_e, raw_n, raw_z = nez_comma_match.group(1), nez_comma_match.group(2), nez_comma_match.group(3)
                        # NEZ 格式順序為 E, N, Z — 根據位數驗證後指派
                        candidates = [raw_e, raw_n, raw_z]
                        coord_n, coord_e, coord_z = None, None, None
                        for c in candidates:
                            int_len = len(c.split('.')[0].lstrip('-'))
                            if int_len == 7 and coord_n is None:
                                coord_n = c
                            elif int_len == 6 and coord_e is None:
                                coord_e = c
                            elif 1 <= int_len <= 4 and coord_z is None:
                                coord_z = c
                        if coord_n and coord_e and coord_z:
                            res["N"] = coord_n
                            res["E"] = coord_e
                            res["H_Z"] = coord_z
                            print(f"  [NEZ逗號格式] N={coord_n}, E={coord_e}, Z={coord_z}")
                            # 直接跳到結尾，不再往下執行
                            pass
                        else:
                            print(f"  [NEZ逗號格式] 找到但位數驗證失敗，繼續往下解析")
                            nez_comma_match = None  # 重設，讓後續邏輯接手

                    # ============================================================
                    # 若 NEZ 逗號格式未成功，執行原有邏輯（使用裁切優先文字）
                    # ============================================================
                    if res["N"] == "N/A" or res["E"] == "N/A" or res["H_Z"] == "N/A":

                        num_matches = list(re.finditer(r'-?\d+(?:\.\d+)?', _c7_text))
                        coord_n, coord_e, coord_z = None, None, None
                        n_match_obj, e_match_obj = None, None

                        # 1. 精準定錨 N（7位整數）和 E（6位整數）
                        for m in num_matches:
                            val_str = m.group()
                            int_digits = len(val_str.split('.')[0].lstrip('-'))
                            if int_digits == 6 and coord_e is None:
                                coord_e = val_str
                                e_match_obj = m
                            elif int_digits == 7 and coord_n is None:
                                coord_n = val_str
                                n_match_obj = m

                        if coord_n and coord_e:
                            res["N"] = coord_n
                            res["E"] = coord_e

                            lines = [line.strip() for line in _c7_text.split('\n') if line.strip()]

                            # ============================================================
                            # 【修正核心】先對全文做「明確標籤定錨」（Z/H/高程），
                            # 不依賴物理行搜尋順序，直接掃描所有行找含 Z 標籤的那行
                            # ============================================================
                            z_label_pattern = re.compile(
                                r'(?:高程|正高|\b[Zz]\s*[:：=\s]|EL\s*[:：=\s])',
                                re.IGNORECASE
                            )
                            # 排除 sigma/精度干擾行的關鍵字（o:、σ:、sigma、PDOP、HDOP 等）
                            sigma_keywords = re.compile(
                                r'\b(?:sigma|pdop|hdop|vdop|rdop|dop)\b|(?:^|\s)[oO]\s*[:：]',
                                re.IGNORECASE
                            )

                            z_found = False

                            # 【第一輪】全行掃描，找有明確 Z/高程標籤的行（跳過 sigma 行）
                            for line in lines:
                                if z_found:
                                    break
                                if sigma_keywords.search(line):
                                    continue
                                z_match = z_label_pattern.search(line)
                                if z_match:
                                    nums_in_line = re.findall(r'-?\d+(?:\.\d+)?', line)
                                    for num in nums_in_line:
                                        if num == coord_n or num == coord_e:
                                            continue
                                        try:
                                            z_val = float(num)
                                        except ValueError:
                                            continue
                                        z_int_len = len(num.split('.')[0].lstrip('-'))
                                        if 1 <= z_int_len <= 4:
                                            coord_z = num
                                            z_found = True
                                            print(f"  [全文標籤定錨成功] Z = {coord_z}（行內容: {line!r}）")
                                            break

                            # 【第二輪】仍未找到 Z，退回物理行鄰近盲猜（排除 sigma 行）
                            if not z_found:
                                target_line_indices = set()
                                for li, line in enumerate(lines):
                                    if coord_n in line or coord_e in line:
                                        target_line_indices.add(li)

                                search_order = []
                                for li in sorted(target_line_indices):
                                    search_order.append(li)
                                    if li + 1 < len(lines): search_order.append(li + 1)
                                    if li + 2 < len(lines): search_order.append(li + 2)
                                    if li - 1 >= 0: search_order.append(li - 1)

                                seen_idx = set()
                                final_order = [i for i in search_order if not (i in seen_idx or seen_idx.add(i))]
                                candidate_lines = [(lines[i], i) for i in final_order if i < len(lines)]

                                for context_line, line_num in candidate_lines:
                                    if z_found:
                                        break
                                    # 跳過 sigma/精度干擾行
                                    if sigma_keywords.search(context_line):
                                        print(f"  [盲猜] 跳過 sigma 干擾行 {line_num}: {context_line!r}")
                                        continue
                                    if any(k in context_line.lower() for k in ['天線', '天线', 'ant', '北', '東', '东']):
                                        context_line = re.sub(
                                            r'.*?(?:天線|天线|ant|antenna)[^0-9]*?\d+(?:\.\d+)?', '',
                                            context_line, flags=re.IGNORECASE
                                        )
                                    line_nums = re.findall(r'-?\d+(?:\.\d+)?', context_line)
                                    for num in line_nums:
                                        if num == coord_n or num == coord_e:
                                            continue
                                        try:
                                            z_val = float(num)
                                        except ValueError:
                                            continue
                                        if abs(z_val) < 1:
                                            continue
                                        z_int_len = len(num.split('.')[0].lstrip('-'))
                                        if 1 <= z_int_len <= 4:
                                            coord_z = num
                                            z_found = True
                                            print(f"  [物理行盲猜成功] 行 {line_num}: 排除 NE 後找到合理數值 Z = {coord_z}")
                                            break

                            # 防錯補位（個位數前字元補位）
                            if coord_z and len(coord_z.split('.')[0]) == 1:
                                z_idx = _c7_text.find(coord_z)
                                if z_idx > 0 and _c7_text[z_idx - 1].isdigit():
                                    coord_z = _c7_text[z_idx - 1] + coord_z
                                    print(f"  [防錯補位] 修正為: {coord_z}")

                            if coord_z:
                                res["H_Z"] = coord_z
                            else:
                                print("  [Class 7 警告] 已識別出 N 和 E，但找不到合理的高程 Z 數值")

                    # 5. 終極 Fallback
                    if any(v == "N/A" for v in [res["N"], res["E"]]):
                        print("  [Class 7 警告] 無法依位數特徵配對出 NE，改採萬用規則。")
                        res = _match_coord(_c7_text, _COORD_PATTERNS)
                elif class_id in _CLASS_PATTERNS:
                    if class_id in [4, 5]:
                        print(f"\n--- [DEBUG Class {class_id} OCR 文字] 檔案: {os.path.basename(fp)} ---")
                        print(text if text else "(完全沒有辨識出任何文字)")
                        print("--------------------------------------------------")
                    res = _match_coord(text, _CLASS_PATTERNS[class_id])
                    if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
                        res = _match_coord(text, _COORD_PATTERNS)
                    # 數字位數 fallback（class 5：北/東標籤被 OCR 吃掉或誤抓非座標數字時，依 TWD97 位數特徵識別）
                    if class_id == 5 and not _validate_twd97(res):
                        fb = _digit_fallback_twd97(text)
                        if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                            res = fb

                    # 數字位數 fallback（class 4：標籤漏讀或抓到不合格的值時，依 TWD97 位數識別）
                    if class_id == 4 and not _validate_twd97(res):
                        _standalone_re = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*m?\s*$')
                        nums_in_order = []
                        for _line in text.split('\n'):
                            _m = _standalone_re.match(_line)
                            if _m:
                                nums_in_order.append(_m.group(1))
                        coord_n, coord_e, coord_z = None, None, None
                        for v in nums_in_order:
                            int_digits = len(v.split('.')[0].lstrip('-'))
                            if int_digits == 7 and coord_n is None:
                                coord_n = v
                            elif int_digits == 6 and coord_n is not None and coord_e is None:
                                coord_e = v
                            elif coord_e is not None and coord_z is None and '.' in v:
                                coord_z = v
                        if coord_n and coord_e and coord_z:
                            res = {"N": coord_n, "E": coord_e, "H_Z": coord_z}
                        # 若獨立行掃不到，改用通用 digit fallback
                        if not _validate_twd97(res):
                            fb = _digit_fallback_twd97(text)
                            if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                                res = fb

                    # class 8（N E Z 純標籤）及其他未特別處理的 class：統一補 digit fallback
                    if class_id not in [4, 5] and not _validate_twd97(res):
                        fb = _digit_fallback_twd97(text)
                        if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                            res = fb
                else:
                    res = _match_coord(text, _COORD_PATTERNS)
                    # class 9（全模式）：pattern 結果不符合 TWD97 也做一次 digit fallback
                    if not _validate_twd97(res):
                        fb = _digit_fallback_twd97(text)
                        if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                            res = fb

                # ── Step C: 這裡修改 needs_review 的輸出內容 ──────────────────
            # 檢查是否有任何一欄欄位為 "N/A"
            # ── Step C: 嚴格格式檢查 ──────────────────────────
            # ── Step C: 嚴格格式檢查 ──────────────────────────
            def _is_valid_format(res):
                # 輔助函式：檢查字串是否包含小數點，且位數正確
                def check_value(val, expected_digits):
                    try:
                        if val == "N/A": return False
                        str_val = str(val)
                        
                        # 1. 檢查是否包含小數點
                        if '.' not in str_val:
                            return False
                        
                        # 2. 計算數字部分（扣除符號與小數點後）的位數
                        int_part = str_val.split('.')[0].lstrip('-')
                        return len(int_part) == expected_digits
                    except:
                        return False
                
                # 條件：N 必須 7 位、E 必須 6 位，且 N, E, Z 都必須包含小數點
                n_valid = check_value(res["N"], 7)
                e_valid = check_value(res["E"], 6)
                
                # 對於 Z (H_Z)，檢查是否有小數點且非 N/A
                z_valid = (res["H_Z"] != "N/A" and '.' in str(res["H_Z"]))
                
                return n_valid and e_valid and z_valid

            if _is_valid_format(res):
                res["needs_review"] = "正常"
            else:
                res["needs_review"] = "請重新拍攝"

            if coord_yolo:
                res["coord_class"] = class_id
            existing[fn] = res
            
        except Exception as e:
            # 發生程式執行錯誤（如圖檔損壞等）也統一填入 "請重新拍攝"
            existing[fn] = {"N": "N/A", "E": "N/A", "H_Z": "N/A",
                            "needs_review": "請重新拍攝", "error": str(e)}
        if idx % 50 == 0 or idx == len(new_items):
            print(f"  [座標OCR] 進度: {idx}/{len(new_items)}")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=4)
    print(f"  [座標OCR] 完成！結果儲存至: {results_path}")

# ─────────────────────────────────────────────────────────────
# STEP 2c  水準點 OCR (PaddleOCR 版)
# ─────────────────────────────────────────────────────────────


def _yolo_crop_regions(img_array, model, conf_thresh, padding):
    """
    用 YOLO 偵測水準點位置，回傳裁切並放大 2x 的 PIL Image 清單。
    依信心值由高到低排序，讓最可能的框優先送 OCR。
    """
    preds = model.predict(source=img_array, verbose=False)[0]
    h_img, w_img = img_array.shape[:2]
    crops = []
    for box in sorted(preds.boxes, key=lambda b: float(b.conf[0]), reverse=True):
        if float(box.conf[0]) < conf_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * padding), int(bh * padding)
        x1c = max(0, x1 - pad_x)
        y1c = max(0, y1 - pad_y)
        x2c = min(w_img, x2 + pad_x)
        y2c = min(h_img, y2 + pad_y)
        crop = Image.fromarray(img_array[y1c:y2c, x1c:x2c])
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC)
        crops.append(crop)
    return crops


def run_benchmark_ocr(output_json_path, folder_path, reader, yolo_model=None):
    """
    增量水準點 OCR（RapidOCR 版）。
    reader     : 已初始化的 RapidOCR 實例（由 main() 傳入）
    yolo_model : 已載入的水準點 YOLO 裁切模型（由 main() 傳入，None = 停用）
    策略零（選用）：YOLO 偵測水準點位置 → 裁切放大 → OCR。
    策略一：全圖辨識。
    策略二：中央 crop 2x 放大（水準點距離較遠時使用），嘗試 30% / 50% 兩種裁切比例。
    """
    out_dir      = os.path.dirname(output_json_path)
    results_path = os.path.join(out_dir, "benchmark_results.json")

    existing = {}
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)

    with open(output_json_path, 'r', encoding='utf-8') as f:
        new_items = [it for it in json.load(f) if it.get("images") and it["images"] not in existing]

    if not new_items:
        print("  [水準點OCR] 無新照片需處理。")
        return

    def _pick_four(result):
        """從 RapidOCR result 取出信心值 > 0.4 的第一個 4 位純數字，回傳 (text, conf)"""
        if not result:
            return "", 0.0
        for ocr_item in result:
            text, conf = ocr_item[1], float(ocr_item[2])
            if conf > 0.4 and len(text) == 4 and text.isdigit():
                return text, round(conf, 4)
        return "", 0.0

    print(f"  [水準點OCR] 開始辨識 {len(new_items)} 張新照片...")

    for idx, item in enumerate(new_items, 1):
        fn = item["images"]
        fp = os.path.join(folder_path, fn)
        try:
            with Image.open(fp) as img:
                img = ImageOps.exif_transpose(img).convert('RGB')
            arr = np.array(img)

            # 策略零：YOLO 偵測 → 裁切放大 → OCR
            number, ocr_conf = "", 0.0
            if yolo_model:
                for crop in _yolo_crop_regions(arr, yolo_model,
                                               BENCHMARK_YOLO_CONF, BENCHMARK_YOLO_PADDING):
                    result, _ = reader(np.array(crop))
                    number, ocr_conf = _pick_four(result)
                    if number:
                        break

            # 策略一：全圖
            if not number:
                result, _ = reader(arr)
                number, ocr_conf = _pick_four(result)

            # 策略二：中央 crop 放大（嘗試 30% 與 50% 兩種範圍）
            if not number:
                w, h = img.size
                for ratio in [0.3, 0.5]:
                    cw, ch = int(w * ratio), int(h * ratio)
                    left, top = (w - cw) // 2, (h - ch) // 2
                    cropped  = img.crop((left, top, left + cw, top + ch))
                    enlarged = cropped.resize((cw * 2, ch * 2), Image.Resampling.BICUBIC)
                    result, _ = reader(np.array(enlarged))
                    number, ocr_conf = _pick_four(result)
                    if number:
                        break

            if number:
                existing[fn] = {"number": number, "ocr_conf": ocr_conf, "needs_review": False}
            else:
                existing[fn] = {"number": "請重新拍攝", "ocr_conf": None, "needs_review": True}
        except Exception as e:
            existing[fn] = {"number": "Error", "ocr_conf": None,
                            "needs_review": True, "error": str(e)}

        if idx % 50 == 0 or idx == len(new_items):
            print(f"  [水準點OCR] 進度: {idx}/{len(new_items)}")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=4)
    print(f"  [水準點OCR] 完成！結果儲存至: {results_path}")
# ─────────────────────────────────────────────────────────────
# 分類模型路由輔助（取代 FOLDER_RULES）
# ─────────────────────────────────────────────────────────────

def _classify_folder_steps(folder_path, model, conf_thresh, class_to_steps):
    """
    從資料夾均勻抽樣最多 3 張圖片，用分類模型預測類別，
    取信心最高的結果決定步驟集合。
    若所有樣本的信心值都低於 conf_thresh，回傳 None → 呼叫端 fallback 到 FOLDER_RULES。
    """
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(_IMG_EXT)]
    if not imgs:
        return None

    # 均勻取最多 3 張（頭、中、尾）
    step = max(1, len(imgs) // 3)
    samples = imgs[::step][:3]

    best_conf, best_steps = 0.0, None
    for fn in samples:
        fp = os.path.join(folder_path, fn)
        try:
            res  = model(fp)[0]
            conf = float(res.probs.top1conf)
            cls  = res.names[res.probs.top1]
            if conf > best_conf:
                best_conf  = conf
                best_steps = class_to_steps.get(cls, [])
        except Exception:
            continue

    if best_conf < conf_thresh or best_steps is None:
        return None
    return set(best_steps)


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def _collect_job_results(job):
    """
    遍歷 job['source_dir'] 下所有 _output 子目錄，
    收集 coord_ocr_results.json / repeat_images.json / suspected_duplicates.json，
    寫出整合後的三支全域 JSON 到 job['output_dir']。
    """
    job_name   = job["job_name"]
    source_dir = job["source_dir"]
    output_dir = job["output_dir"]

    global_all_results  = {}
    global_repeat       = []
    global_suspected    = {}

    for root, dirs, _ in os.walk(source_dir):
        dirs[:] = [d for d in dirs if not d.endswith("_output")]
        folder_name = os.path.basename(root)
        out_dir = os.path.join(root, f"{job_name}{folder_name}_output")
        if not os.path.exists(out_dir):
            continue

        # 座標 OCR 結果
        coord_path = os.path.join(out_dir, "coord_ocr_results.json")
        if os.path.exists(coord_path):
            try:
                with open(coord_path, 'r', encoding='utf-8') as f:
                    global_all_results.update(json.load(f))
            except Exception as e:
                print(f"讀取 {coord_path} 整合時發生錯誤: {e}")

        # 水準點 OCR 結果（若存在也一併整合）
        bench_path = os.path.join(out_dir, "benchmark_results.json")
        if os.path.exists(bench_path):
            try:
                with open(bench_path, 'r', encoding='utf-8') as f:
                    global_all_results.update(json.load(f))
            except Exception as e:
                print(f"讀取 {bench_path} 整合時發生錯誤: {e}")

        # 重複清單（Bug 1 修正：repeat_images.json 是 dict 格式）
        repeat_path = os.path.join(out_dir, "repeat_images.json")
        if os.path.exists(repeat_path):
            try:
                with open(repeat_path, 'r', encoding='utf-8') as f:
                    local_repeat = json.load(f)
                if isinstance(local_repeat, dict):
                    for fn in local_repeat:
                        global_repeat.append({"images": fn})
                elif isinstance(local_repeat, list):
                    for item in local_repeat:
                        fn = item.get("images") if isinstance(item, dict) else item
                        if fn:
                            global_repeat.append({"images": fn})
            except Exception as e:
                print(f"讀取 {repeat_path} 整合時發生錯誤: {e}")

        # 疑似重複清單
        if PHASH_SUSPECTED_ENABLED:
            suspected_path = os.path.join(out_dir, "suspected_duplicates.json")
            if os.path.exists(suspected_path):
                try:
                    with open(suspected_path, 'r', encoding='utf-8') as f:
                        global_suspected.update(json.load(f))
                except Exception as e:
                    print(f"讀取 {suspected_path} 整合時發生錯誤: {e}")

    os.makedirs(output_dir, exist_ok=True)

    all_path    = os.path.join(output_dir, f"{job_name}_all_results.json")
    repeat_path = os.path.join(output_dir, f"{job_name}_repeat_images.json")
    with open(all_path, 'w', encoding='utf-8') as f:
        json.dump(global_all_results, f, ensure_ascii=False, indent=4)
    print(f"  -> OCR 結果: {len(global_all_results)} 筆 → {all_path}")

    global_repeat = sorted(global_repeat, key=lambda x: x["images"])
    with open(repeat_path, 'w', encoding='utf-8') as f:
        json.dump(global_repeat, f, ensure_ascii=False, indent=4)
    print(f"  -> 重複清單: {len(global_repeat)} 筆 → {repeat_path}")

    if PHASH_SUSPECTED_ENABLED and global_suspected:
        susp_path = os.path.join(output_dir, f"{job_name}_suspected_duplicates.json")
        with open(susp_path, 'w', encoding='utf-8') as f:
            json.dump(global_suspected, f, ensure_ascii=False, indent=4)
        print(f"  -> 疑似重複: {len(global_suspected)} 筆 → {susp_path}")


def main():
    # ── 一次性初始化所有需要的模型 ────────────────────────────
    all_steps = set(s for job in JOBS for s in job.get("steps", []))

    ocr_engine = None
    if "coord_ocr" in all_steps or "benchmark" in all_steps:
        print("[初始化] 載入 RapidOCR...")
        from rapidocr_onnxruntime import RapidOCR
        ocr_engine = RapidOCR()

    coord_yolo  = None
    yolo_device = 'cpu'
    if "coord_ocr" in all_steps and COORD_YOLO_ENABLED:
        from ultralytics import YOLO as YOLOModel
        try:
            coord_yolo  = YOLOModel(COORD_YOLO_MODEL_PATH).to('cuda')
            yolo_device = 'cuda'
            print(f"[初始化] 格式分類模型載入成功 (GPU): {COORD_YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[初始化] GPU 載入失敗 ({e})，改用 CPU...")
            try:
                coord_yolo  = YOLOModel(COORD_YOLO_MODEL_PATH).to('cpu')
                yolo_device = 'cpu'
                print(f"[初始化] 格式分類模型載入成功 (CPU): {COORD_YOLO_MODEL_PATH}")
            except Exception as cpu_e:
                print(f"[初始化] CPU 亦失敗: {cpu_e}，格式分類停用")

    benchmark_yolo = None
    if "benchmark" in all_steps and BENCHMARK_YOLO_CROP:
        from ultralytics import YOLO as YOLOModel
        try:
            benchmark_yolo = YOLOModel(BENCHMARK_YOLO_MODEL_PATH)
            print(f"[初始化] 水準點裁切模型載入成功: {BENCHMARK_YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[初始化] 水準點裁切模型載入失敗: {e}")

    ruler_yc_model = None
    if "ruler_yc" in all_steps and RULER_YC_ENABLED:
        from ultralytics import YOLO as YOLOModel
        try:
            ruler_yc_model = YOLOModel(RULER_YC_MODEL_PATH)
            print(f"[初始化] 尺規YC模型載入成功: {RULER_YC_MODEL_PATH}")
        except Exception as e:
            print(f"[初始化] 尺規YC模型載入失敗: {e}，ruler_yc 停用")

    classify_model = None
    if USE_CLASSIFY_MODEL:
        from ultralytics import YOLO as YOLOModel
        try:
            classify_model = YOLOModel(CLASSIFY_MODEL_PATH)
            print(f"[初始化] 資料夾分類模型載入成功: {CLASSIFY_MODEL_PATH}")
        except Exception as e:
            print(f"[初始化] 資料夾分類模型載入失敗: {e}，回退到 FOLDER_RULES")

    # ── 逐一執行每個 job ──────────────────────────────────────
    for job in JOBS:
        source_dir = job["source_dir"]
        job_name   = job["job_name"]
        job_steps  = set(job.get("steps", []))

        if not os.path.exists(source_dir):
            print(f"\n[跳過] 找不到目錄: {source_dir}")
            continue

        print(f"\n{'='*10} {job_name} {'='*10}")
        print(f"  來源: {source_dir}")

        for root, dirs, _ in os.walk(source_dir):
            dirs[:] = [d for d in dirs if not d.endswith("_output")]
            folder_name = os.path.basename(root)

            # Step 1: 去重
            output_json = dedup_folder(root, folder_name, job_name)
            if not output_json:
                continue

            # ── 步驟路由：分類模型優先，fallback 到 FOLDER_RULES ──
            steps = set()
            if classify_model:
                steps = _classify_folder_steps(root, classify_model,
                                               CLASSIFY_CONF_THRESH, CLASSIFY_CLASS_STEPS) or set()
                if steps:
                    print(f"  [路由-分類模型] 步驟: {sorted(steps)}")
                else:
                    print(f"  [路由-分類模型] 信心不足，使用 job steps")

            if not steps:
                if job_steps:
                    # JOBS 直接指定步驟時優先使用
                    steps = job_steps
                    print(f"  [路由-job設定] 步驟: {sorted(steps)}")
                else:
                    # fallback 到 FOLDER_RULES 關鍵字比對
                    match_target = folder_name + "|" + root
                    for keyword, kw_steps in FOLDER_RULES.items():
                        if keyword in match_target:
                            steps.update(kw_steps)
                    if not steps:
                        if NO_MATCH_DEDUPLICATE_SWITCH == "OPEN":
                            print(f"  [路由-未命中] OPEN：僅去重")
                        else:
                            print(f"  [路由-未命中] CLOSE：跳過")
                            continue
                    else:
                        print(f"  [路由-關鍵字] 步驟: {sorted(steps)}")

            # Step 2a: YOLO 尺規辨識
            if "yolo" in steps and RULER_YOLO_ENABLED:
                run_yolo(output_json, root, YOLO_MODEL_PATH)

            # Step 2a-alt: A+C 尺規辨識
            if "ruler_ac" in steps:
                run_ruler_ac(output_json, root)

            # Step 2a-yc: YOLO垂直尺 + 接觸偵測
            if "ruler_yc" in steps and RULER_YC_ENABLED and ruler_yc_model is not None:
                run_ruler_yolo_contact(output_json, root, RULER_YC_MODEL_PATH)

            # Step 2b: 座標 OCR
            if "coord_ocr" in steps:
                run_coord_ocr(output_json, root, ocr_engine, coord_yolo)

            # Step 2c: 水準點 OCR
            if "benchmark" in steps:
                run_benchmark_ocr(output_json, root, ocr_engine, benchmark_yolo)

        # ── 全域整合輸出（每個 job 獨立） ─────────────────────
        print(f"\n[整合] {job_name}...")
        _collect_job_results(job)

    print("\n============ 所有任務完成 ============")


if __name__ == "__main__":
    main()
