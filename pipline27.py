"""
pipeline_0525.py — 影像去重與多模態辨識整合管道

工作流程（每個資料夾獨立執行）:
  Step 1  去重     : pHash 比對去重，產生 output.json / repeat_images.json / corrupted_images.json
  Step 2a 尺規     : YOLO 辨識直/橫尺是否交叉，產生 yolo_results.json
  Step 2b 座標OCR  : RapidOCR OCR 辨識 N/E/H 座標，產生 coord_ocr_results.json
  Step 2c 水準點   : RapidOCR 辨識 4 位數水準點號碼，產生 benchmark_results.json

增量設計:
  - 去重  : 新照片才計算 hash，已知照片沿用 hashes.json 的紀錄
  - 辨識  : 只對 output.json 中尚未有結果的照片處理，結果持續附加至各 JSON
"""
import os
import json
import re
import numpy as np
from PIL import Image, ImageOps
import imagehash

# =============================================
# 設定區 — 請依實際情況修改
# =============================================
BASE_DIR = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\112"
YEARS    = ["驗證測量讀數照片"]

# ── 工具路徑 ──────────────────────────────────
YOLO_MODEL_PATH    = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\best.pt"
RULER_YOLO_ENABLED = True   # False = 只跑 ruler_ac（A+C法），跳過 YOLO 尺規辨識
TESSERACT_CMD      = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
COORD_OCR_GROUP_ID = None  # 已改用 PaddleOCR，此參數保留相容性，不再生效
BENCHMARK_GPU      = True

# ── 水準點 YOLO 輔助裁切（訓練完模型後將 BENCHMARK_YOLO_CROP 改為 True）──
BENCHMARK_YOLO_CROP       = True                        # False = 停用；True = 啟用
BENCHMARK_YOLO_MODEL_PATH = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\runs\detect\train-7\weights\best.pt"# 訓練好的一等水準點偵測模型
BENCHMARK_YOLO_CONF       = 0.25   # 偵測信心閾值（低一點避免漏偵測）
BENCHMARK_YOLO_PADDING    = 0.3    # 裁切框四周額外留邊比例（相對於框的寬/高）

# ── A+C 尺規辨識參數 ───────────────────────────
# 垂直黃色尺的 HSV 範圍（可用 cv2 取樣後微調）
YELLOW_HSV_LOWER = [18, 100, 100]
YELLOW_HSV_UPPER = [38, 255, 255]
# 水平橫條最小長度（佔圖寬比例），過短的線段視為背景雜訊
HOUGH_MIN_LINE_RATIO = 0.35

# ── 步驟路由 ──────────────────────────────────
# 資料夾名稱（或路徑）包含 key 時，執行 value 中的步驟。
# 步驟代號: "yolo" | "ruler_ac" | "coord_ocr" | "benchmark"
# 多個關鍵字可同時命中，步驟取聯集。
FOLDER_RULES = {
    # "埋深":  ["ruler_ac", "yolo"],
    # "水準點": ["benchmark"],
    # "測量讀數照片": ["coord_ocr"],
    # "測量坐標讀數": ["coord_ocr"],
    # "測量拍照-測量照片讀數照片": ["coord_ocr"],
    # "測量照片讀數-照片": ["coord_ocr"],
    # "坐標測量": ["coord_ocr"],

    #本周先執行(0527~0529)
    "測試": ["coord_ocr"],
    
}
# 無關鍵字命中時的預設步驟（空清單 = 只做去重）
DEFAULT_STEPS = []

# ── 影像分類模型路由（訓練完模型後將 USE_CLASSIFY_MODEL 改為 True）──────
USE_CLASSIFY_MODEL   = False                      # False = 使用 FOLDER_RULES；True = 使用分類模型
CLASSIFY_MODEL_PATH  = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\0526OCR\classify_output\runs\classify\weights\best.pt" # classify_train.py 訓練輸出的 best.pt
CLASSIFY_CONF_THRESH = 0    # 低於此信心值 → 自動 fallback 到 FOLDER_RULES 0.6
# 分類類別名稱 → 步驟（key 必須與 classify_train.py 的 CLASS_DIRS key 一致）
CLASSIFY_CLASS_STEPS = {
    "埋深":   ["ruler_ac", "yolo"],
    "水準點": ["benchmark"],
    "座標":   ["coord_ocr"],
    "其他":   [],
}

# ── OCR 信心分數閾值 ────────────────────────────────
OCR_LOW_CONF_THRESH = 0.6   # 座標 OCR：低於此比例的欄位為 N/A 時標記為需複核
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


def dedup_folder(folder_path, folder_name, year_prefix):
    """
    對 folder_path 內的照片進行增量 pHash 去重。
    已知照片從 hashes.json 讀取，不重複計算；新照片才計算並比對。
    回傳 output.json 路徑；資料夾無照片則回傳 None。
    """
    all_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(_IMG_EXT) and os.path.isfile(os.path.join(folder_path, f))
    )
    if not all_files:
        return None

    out_dir = os.path.join(folder_path, f"{year_prefix}{folder_name}_output")
    os.makedirs(out_dir, exist_ok=True)

    remain_path  = os.path.join(out_dir, "output.json")
    dup_path     = os.path.join(out_dir, "repeat_images.json")
    corrupt_path = os.path.join(out_dir, "corrupted_images.json")
    hashes_path  = os.path.join(out_dir, "hashes.json")

    known_remain  = _load_json_set(remain_path)
    known_dup     = _load_json_set(dup_path)
    known_corrupt = _load_json_set(corrupt_path)
    known_all     = known_remain | known_dup | known_corrupt

    saved_hashes = {}
    if os.path.exists(hashes_path):
        with open(hashes_path, 'r', encoding='utf-8') as f:
            saved_hashes = json.load(f)

    new_files = [f for f in all_files if f not in known_all]
    if not new_files:
        print(f"  [{folder_path}] 無新照片，跳過去重。")
        return remain_path if os.path.exists(remain_path) else None

    print(f"\n正在去重: {folder_path}（新照片 {len(new_files)} 張）")

    # 重建已不重複照片的 hash → filename 對應
    existing_hashes = {
        imagehash.hex_to_hash(saved_hashes[fn]): fn
        for fn in known_remain if fn in saved_hashes
    }

    new_remain, new_dup, new_corrupt = set(), set(), set()
    for fn in new_files:
        fp = os.path.join(folder_path, fn)
        try:
            with Image.open(fp) as img:
                h = imagehash.phash(img)
            if h in existing_hashes:
                new_dup.add(fn)
            else:
                existing_hashes[h] = fn
                saved_hashes[fn] = str(h)
                new_remain.add(fn)
        except Exception as e:
            print(f"  [錯誤] {fn}: {e}")
            new_corrupt.add(fn)

    final_remain  = known_remain  | new_remain
    final_dup     = known_dup     | new_dup
    final_corrupt = known_corrupt | new_corrupt

    _save_json_list(remain_path, final_remain)
    _save_json_list(dup_path, final_dup)
    if final_corrupt:
        _save_json_list(corrupt_path, final_corrupt)

    with open(hashes_path, 'w', encoding='utf-8') as f:
        json.dump(saved_hashes, f, ensure_ascii=False, indent=4)

    print(f"  -> 不重複: {len(final_remain)} | 重複: {len(final_dup)} | 損毀: {len(final_corrupt)}")
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
                "total_images", "both_detected_crossed", "both_detected_not_crossed",
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
        for box in preds.boxes:
            if float(box.conf[0].item()) > 0.05:
                xyxy = box.xyxy[0].tolist()
                (vb if int(box.cls[0].item()) == 1 else hb).append(xyxy)

        vc, hc = len(vb), len(hb)
        if vc == 0 and hc == 0:
            status = "None Detected";               sm["none_detected"] += 1
        elif vc > 0 and hc == 0:
            status = "Vertical Only";               sm["vertical_only"] += 1
        elif vc == 0 and hc > 0:
            status = "Horizontal Only";             sm["horizontal_only"] += 1
        elif any(_bbox_crossed(b1, b2) for b1 in vb for b2 in hb):
            status = "Both Detected (Crossed)";     sm["both_detected_crossed"] += 1
        else:
            status = "Both Detected (Not Crossed)"; sm["both_detected_not_crossed"] += 1

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

def _detect_rulers_ac(img_path):
    """
    偵測單張圖片中的垂直黃色直尺（C法）與水平橫條（A法）。
    回傳 (has_vertical, has_horizontal, is_crossed, v_boxes, h_lines)
    """
    import cv2

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False, False, False, [], []

    h_img, w_img = img.shape[:2]

    # ── C: HSV 黃色遮罩 → 垂直尺 ─────────────────────
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array(YELLOW_HSV_LOWER, dtype=np.uint8),
                       np.array(YELLOW_HSV_UPPER, dtype=np.uint8))
    # 閉運算填補黑色刻度造成的斷點
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    v_boxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 400:
            continue
        # x, y, cw, ch = cv2.boundingRect(c)
        # if ch / max(cw, 1) > 2.5 and ch > h_img * 0.12:  # 長寬比 > 2.5 且夠長
        #     v_boxes.append((x, y, x + cw, y + ch))
        # ── 取代原本 boundingRect 的寫法 ──
        rect = cv2.minAreaRect(c)  # 得到 ((cx, cy), (w, h), angle)
        (cw, ch) = rect[1]
        # 確保 ch 永遠是長邊，cw 是短邊
        if cw > ch:
            cw, ch = ch, cw

        if ch / max(cw, 1) > 2.5 and ch > h_img * 0.12:
            # 為了維持原程式格式，這裡計算其包覆範圍，但長寬比已經用旋轉矩形驗證過了
            x, y, w, h = cv2.boundingRect(c)
            v_boxes.append((x, y, x + w, y + h))

    has_vertical = len(v_boxes) > 0

    # ── A: Hough 直線 → 水平橫條 ─────────────────────
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_len = int(w_img * HOUGH_MIN_LINE_RATIO)
    lines   = cv2.HoughLinesP(edges, 1, np.pi / 180,
                               threshold=60, minLineLength=min_len, maxLineGap=25)
    h_lines = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 35 or angle > 145:   # ±20° 視為水平
                h_lines.append((x1, y1, x2, y2))

    has_horizontal = len(h_lines) > 0

    # ── 交叉判斷 ─────────────────────────────────────
    is_crossed = False
    if has_vertical and has_horizontal:
        for hx1, hy1, hx2, hy2 in h_lines:
            hcy   = (hy1 + hy2) / 2
            hx_lo = min(hx1, hx2)
            hx_hi = max(hx1, hx2)
            for vx1, vy1, vx2, vy2 in v_boxes:
                vcx = (vx1 + vx2) / 2
                if (vy1 <= hcy <= vy2) and (hx_lo <= vcx <= hx_hi):
                    is_crossed = True
                    break
            if is_crossed:
                break

    return has_vertical, has_horizontal, is_crossed, v_boxes, h_lines


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
                "total_images", "both_detected_crossed", "both_detected_not_crossed",
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
            status = "None Detected";               sm["none_detected"] += 1
        elif has_v and not has_h:
            status = "Vertical Only";               sm["vertical_only"] += 1
        elif not has_v and has_h:
            status = "Horizontal Only";             sm["horizontal_only"] += 1
        elif crossed:
            status = "Both Detected (Crossed)";     sm["both_detected_crossed"] += 1
        else:
            status = "Both Detected (Not Crossed)"; sm["both_detected_not_crossed"] += 1

        rec.update(status=status, detections={"has_vertical": has_v, "has_horizontal": has_h})
        report["results_list"].append(rec)

        if idx % 100 == 0 or idx == len(new_items):
            print(f"  [尺規A+C] 進度: {idx}/{len(new_items)}")

    sm["total_images"] = len(report["results_list"])
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"  [尺規A+C] 完成！結果儲存至: {results_path}")


# ─────────────────────────────────────────────────────────────
# STEP 2b  座標 OCR (orgocr 邏輯)
# ─────────────────────────────────────────────────────────────

# # 基礎數字正則：支援正負號、小數點
# _NUM = r'(-?\d+\.\d+|-?\d+)'

# # 依優先順序排列的座標 pattern，每組為 (N_regex, E_regex, H_regex)。
# # 具體標籤（北座標、地表N）排前，避免通用字母（N/E/H）誤匹配。
# _COORD_PATTERNS = [
#     # 【新增：優化 RTK App 截圖格式】
#     # 針對單一字母 N, E, Z 獨立成行的精準匹配（使用斷言排除 σN, σE, σZ 與前後干擾，且限定同分行 [ \t]）
#     (r'(?<![a-zA-Z0-9σ])N[ \t]*' + _NUM, r'(?<![a-zA-Z0-9σ])E[ \t]*' + _NUM, r'(?<![a-zA-Z0-9σ])Z[ \t]*' + _NUM),
    
#     # 【原始 Pattern 家族】
#     (r'北座標[:=\s]*'     + _NUM, r'東座標[:=\s]*'     + _NUM, r'高程[:=\s]*'         + _NUM),
#     (r'縱軸[:=\s]*'        + _NUM, r'橫軸[:=\s]*'        + _NUM, r'高程[:=\s]*'         + _NUM),
#     (r'地表.*?[NN][:=\s]*' + _NUM, r'地表.*?[EE][:=\s]*' + _NUM, r'地表.*?[HH][:=\s]*' + _NUM),
#     (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'本地.*?[HH][:=\s]*' + _NUM),
#     (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'高程[:=\s]*'         + _NUM),
#     (r'北[:=\s]*'          + _NUM, r'東[:=\s]*'          + _NUM, r'高程?[:=\s]*'        + _NUM),
#     (r'北[:=\s]*'          + _NUM, r'東[:=\s]*'          + _NUM, r'高度[:=\s]*'         + _NUM),
#     (r'[NN][:=\s]*'        + _NUM, r'[EE][:=\s]*'        + _NUM, r'[ZZ][:=\s]*'         + _NUM),
#     (r'[NN][:=\s]*'        + _NUM, r'[EE][:=\s]*'        + _NUM, r'[HH][:=\s]*'         + _NUM),
# ]



# def _ocr_with_paddle(image_path, engine):
#     """RapidOCR 辨識，回傳合併文字（信心值 > 0.3 的結果）"""
#     result, _ = engine(image_path)
#     if not result:
#         return ""
#     return "\n".join(item[1] for item in result if float(item[2]) > 0.3)


# def _extract_coord(image_path, engine):
#     """PaddleOCR 辨識座標，依序嘗試 _COORD_PATTERNS 直到三項全中。"""
#     text = _ocr_with_paddle(image_path, engine)
#     best = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}

#     # 特殊格式: NEZ,N,E,Z
#     m = re.search(r'NEZ[:=\s]*' + _NUM + r',' + _NUM + r',' + _NUM, text.replace(' ', ''))
#     if m:
#         return {"N": m.group(1), "E": m.group(2), "H_Z": m.group(3)}

#     for np_, ep_, hp_ in _COORD_PATTERNS:
#         mn = re.search(np_, text, re.IGNORECASE)
#         me = re.search(ep_, text, re.IGNORECASE)
#         mh = re.search(hp_, text, re.IGNORECASE)
#         if mn and me and mh:
#             return {"N": mn.group(1), "E": me.group(1), "H_Z": mh.group(1)}
#         # 部分命中：保留已找到的欄位，繼續下一個 pattern
#         if mn and best["N"]   == "N/A": best["N"]   = mn.group(1)
#         if me and best["E"]   == "N/A": best["E"]   = me.group(1)
#         if mh and best["H_Z"] == "N/A": best["H_Z"] = mh.group(1)

#     return best


# def run_coord_ocr(output_json_path, folder_path, group_id=None):
#     """
#     增量座標 OCR（PaddleOCR 版）。
#     group_id 參數保留相容性，不再使用。
#     """
#     out_dir      = os.path.dirname(output_json_path)
#     results_path = os.path.join(out_dir, "coord_ocr_results.json")

#     existing = {}
#     if os.path.exists(results_path):
#         with open(results_path, 'r', encoding='utf-8') as f:
#             existing = json.load(f)

#     with open(output_json_path, 'r', encoding='utf-8') as f:
#         new_items = [it for it in json.load(f) if it.get("images") and it["images"] not in existing]

#     if not new_items:
#         print("  [座標OCR] 無新照片需處理。")
#         return

#     print("  [座標OCR] 初始化 RapidOCR...")
#     from rapidocr_onnxruntime import RapidOCR
#     engine = RapidOCR()
#     print(f"  [座標OCR] 開始辨識 {len(new_items)} 張新照片...")
#     for idx, item in enumerate(new_items, 1):
#         fn = item["images"]
#         fp = os.path.join(folder_path, fn)
#         try:
#             res = _extract_coord(fp, engine)
#             res["needs_review"] = any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]])
#             existing[fn] = res
#         except Exception as e:
#             existing[fn] = {"N": "N/A", "E": "N/A", "H_Z": "N/A",
#                             "needs_review": True, "error": str(e)}
#         if idx % 50 == 0 or idx == len(new_items):
#             print(f"  [座標OCR] 進度: {idx}/{len(new_items)}")

#     with open(results_path, 'w', encoding='utf-8') as f:
#         json.dump(existing, f, ensure_ascii=False, indent=4)
#     print(f"  [座標OCR] 完成！結果儲存至: {results_path}")


# ─────────────────────────────────────────────────────────────
# STEP 2b  座標 OCR (orgocr 邏輯) - 完美修正完整版
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# STEP 2b  座標 OCR (orgocr 邏輯) - 終極特徵驗證版
# ─────────────────────────────────────────────────────────────


# 基礎數字正則：支援正負號、小數點
_NUM = r'(-?\d+\.\d+|-?\d+)'

# 依優先順序排列的座標一體化 pattern，每組為 (N_regex, E_regex, H_regex)。
_COORD_PATTERNS = [
    (r'地表\s*N[^-\d]*' + _NUM, r'地表\s*E[^-\d]*' + _NUM, r'地表\s*H[^-\d]*' + _NUM),
    (r'本地\s*N[^-\d]*' + _NUM, r'本地\s*E[^-\d]*' + _NUM, r'本地\s*H[^-\d]*' + _NUM),
    (r'本地\s*N[^-\d]*' + _NUM, r'本地\s*E[^-\d]*' + _NUM, r'高程[:=\s]*'         + _NUM),
    (r'北座標[:=\s]*'     + _NUM, r'東座標[:=\s]*'     + _NUM, r'高程[:=\s]*'         + _NUM),
    (r'縱軸[:=\s]*'        + _NUM, r'橫軸[:=\s]*'        + _NUM, r'高程[:=\s]*'         + _NUM),
    (r'北[:=\s]*'          + _NUM, r'東[:=\s]*'          + _NUM, r'高程?[:=\s]*'        + _NUM),
    (r'北[:=\s]*'          + _NUM, r'東[:=\s]*'          + _NUM, r'高度[:=\s]*'         + _NUM),
    (r'(?<![a-zA-Z0-9σ])N[ \t]*' + _NUM, r'(?<![a-zA-Z0-9σ])E[ \t]*' + _NUM, r'(?<![a-zA-Z0-9σ])Z[ \t]*' + _NUM),
    (r'[NN][:=\s]*'        + _NUM, r'[EE][:=\s]*'        + _NUM, r'[ZZ][:=\s]*'         + _NUM),
    (r'[NN][:=\s]*'        + _NUM, r'[EE][:=\s]*'        + _NUM, r'[HH][:=\s]*'         + _NUM),
]


def _ocr_with_paddle(image_path, engine):
    """RapidOCR 辨識，回傳合併文字（信心值 > 0.3 的結果）"""
    result, _ = engine(image_path)
    if not result:
        return ""
    return "\n".join(item[1] for item in result if float(item[2]) > 0.3)


def _is_valid_n(val):
    """驗證是否為合法的台灣二度分帶北座標 (7位數，且多以25、26開頭)"""
    try:
        s = val.split('.')[0].replace('-', '').strip()
        return len(s) == 7 and s.startswith(('25', '26'))
    except:
        return False


def _is_valid_e(val):
    """驗證是否為合法的台灣二度分帶東座標 (6位數，通常在15萬~35萬之間)"""
    try:
        s = val.split('.')[0].replace('-', '').strip()
        if len(s) == 6:
            prefix = int(s[:2])
            return 15 <= prefix <= 35
        return False
    except:
        return False


def _extract_coord(image_path, engine):
    """
    特徵物理驗證版座標擷取：不論用什麼方法抓，都必須符合台灣 TWD97 座標數學特徵。
    """
    text = _ocr_with_paddle(image_path, engine)
    best = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}

    if not text:
        return best

    # 預清洗：按行切分
    raw_lines = [line.strip() for line in text.split('\n') if line.strip()]
    cleaned_lines = []
    
    # 過濾包含時間、經緯度度分秒、或明確含有標準差符號 σ 的行
    for line in raw_lines:
        if 'σ' in line or 'oN' in line or 'oE' in line or 'oZ' in line:
            continue
        if (':' in line or '°' in line or "'" in line) and ('座標' not in line and '地表' not in line and '本地' not in line and '北' not in line and '東' not in line):
            continue
        cleaned_lines.append(line)
        
    reconstructed_text = "\n".join(cleaned_lines)

    # =================================================================
    # 【第一階段】傳統一體化 Pattern 比對（通過嚴格數學特徵過濾）
    # =================================================================
    # 特殊格式處理 NEZ: N,E,Z
    clean_text = re.sub(r'\s+', '', reconstructed_text)
    m = re.search(r'NEZ[:=\s]*' + _NUM + r',' + _NUM + r',' + _NUM, clean_text, re.IGNORECASE)
    if m and _is_valid_n(m.group(1)) and _is_valid_e(m.group(2)):
        best["N"], best["E"], best["H_Z"] = m.group(1), m.group(2), m.group(3)
        return best

    for np_, ep_, hp_ in _COORD_PATTERNS:
        mn = re.search(np_, reconstructed_text, re.IGNORECASE)
        me = re.search(ep_, reconstructed_text, re.IGNORECASE)
        mh = re.search(hp_, reconstructed_text, re.IGNORECASE)
        
        # 抓取並驗證 N
        if mn and best["N"] == "N/A" and _is_valid_n(mn.group(1)):
            best["N"] = mn.group(1)
        # 抓取並驗證 E
        if me and best["E"] == "N/A" and _is_valid_e(me.group(1)):
            best["E"] = me.group(1)
        # 抓取 H_Z
        if mh and best["H_Z"] == "N/A":
            best["H_Z"] = mh.group(1)

        if all(v != "N/A" for v in best.values()):
            return best

    # =================================================================
    # 【第二階段】逐行精細掃描（由後往前，專治 3174_9、4581_81、與高程錯位）
    # =================================================================
    for line in reversed(cleaned_lines):
        if best["N"] == "N/A":
            nm = re.search(r'(?:地表\s*N|本地\s*N|北座標|(?<![a-zA-Z0-9])N\b|(?<![a-zA-Z0-9])北)[^-\d]*' + _NUM, line, re.IGNORECASE)
            if nm and _is_valid_n(nm.group(1)):
                best["N"] = nm.group(1)
                
        if best["E"] == "N/A":
            em = re.search(r'(?:地表\s*E|本地\s*E|東座標|(?<![a-zA-Z0-9])E\b|(?<![a-zA-Z0-9])東)[^-\d]*' + _NUM, line, re.IGNORECASE)
            if em and _is_valid_e(em.group(1)):
                best["E"] = em.group(1)
                
        if best["H_Z"] == "N/A":
            # 高程通常在大地高/橢球高的下方，由下往上抓可以確保抓到主表格內真實高程
            hm = re.search(r'(?:地表\s*H|本地\s*H|高程|(?<![a-zA-Z0-9])[ZH]\b)[^-\d]*' + _NUM, line, re.IGNORECASE)
            if hm:
                val = hm.group(1)
                # 排除被 N/E 佔用的數字，且過濾掉明顯小於 0.05 的標準差雜訊（防 4805_160 錯誤）
                if val != best["N"] and val != best["E"]:
                    try:
                        if abs(float(val)) > 0.05:
                            best["H_Z"] = val
                    except:
                        best["H_Z"] = val

    # =================================================================
    # 【第三階段】數學特徵大保底（專治 3239_23、6845_1858 等分欄表格完全拆家狀況）
    # =================================================================
    if best["N"] == "N/A" or best["E"] == "N/A":
        all_numbers = re.findall(r'-?\d+\.\d+|-?\d+', reconstructed_text)
        for cand in all_numbers:
            if best["N"] == "N/A" and _is_valid_n(cand):
                best["N"] = cand
            elif best["E"] == "N/A" and _is_valid_e(cand):
                best["E"] = cand

    # =================================================================
    # 【第四階段】高程 H_Z 落單與重複標籤保底（防 4646_104 狀態列搶占）
    # =================================================================
    if best["H_Z"] == "N/A":
        # 從最後一行往前尋找最靠近底部的合理高程數字
        for line in reversed(cleaned_lines):
            if any(k in line for k in ['高程', '高', 'H', 'Z', 'h', 'z']):
                nums_in_line = re.findall(r'-?\d+\.\d+|-?\d+', line)
                for n in nums_in_line:
                    if n != best["N"] and n != best["E"]:
                        try:
                            # 座標數據整數通常長度小於 5 位數，且排除小於 0.05 的標準差
                            if len(n.split('.')[0].replace('-', '')) <= 4 and abs(float(n)) > 0.05:
                                best["H_Z"] = n
                                break
                        except:
                            best["H_Z"] = n
                            break
            if best["H_Z"] != "N/A":
                break

    return best


def run_coord_ocr(output_json_path, folder_path, group_id=None):
    """
    增量座標 OCR（PaddleOCR 版）。
    group_id 參數保留相容性，不再使用。
    """
    out_dir      = os.path.dirname(output_json_path)
    if not out_dir:
        out_dir = "."
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

    print("  [座標OCR] 初始化 RapidOCR...")
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    print(f"  [座標OCR] 開始辨識 {len(new_items)} 張新照片...")
    for idx, item in enumerate(new_items, 1):
        fn = item["images"]
        fp = os.path.join(folder_path, fn)
        try:
            res = _extract_coord(fp, engine)
            res["needs_review"] = any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]])
            existing[fn] = res
        except Exception as e:
            existing[fn] = {"N": "N/A", "E": "N/A", "H_Z": "N/A",
                            "needs_review": True, "error": str(e)}
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


def run_benchmark_ocr(output_json_path, folder_path):
    """
    增量水準點 OCR（RapidOCR 版）。
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

    print("  [水準點OCR] 初始化 RapidOCR...")
    from rapidocr_onnxruntime import RapidOCR
    reader = RapidOCR()

    # ── 策略零：YOLO 輔助裁切（需在設定區啟用）──────────────
    yolo_model = None
    if BENCHMARK_YOLO_CROP:
        from ultralytics import YOLO as YOLOModel
        yolo_model = YOLOModel(BENCHMARK_YOLO_MODEL_PATH)
        print(f"  [水準點OCR] 載入 YOLO 裁切模型: {BENCHMARK_YOLO_MODEL_PATH}")

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

def main():
    for year in YEARS:
        year_dir = os.path.join(BASE_DIR, year)
        if not os.path.exists(year_dir):
            print(f"找不到年份資料夾: {year_dir}，跳過。")
            continue

        print(f"\n========== {year} 年度 ==========")

        # ── 分類模型初始化（每個年度只載入一次）──────────────
        classify_model = None
        if USE_CLASSIFY_MODEL:
            from ultralytics import YOLO as YOLOModel
            try:
                classify_model = YOLOModel(CLASSIFY_MODEL_PATH)
                print(f"  [分類模型] 已載入: {CLASSIFY_MODEL_PATH}")
            except Exception as e:
                print(f"  [分類模型] 載入失敗: {e}，回退到 FOLDER_RULES")

        for root, dirs, _ in os.walk(year_dir):
            dirs[:] = [d for d in dirs if not d.endswith("_output")]
            folder_name = os.path.basename(root)
            if folder_name == year:
                folder_name = f"{year}年度"

            # Step 1: 去重，產生 output.json
            output_json = dedup_folder(root, folder_name, year)
            if not output_json:
                continue

            # ── 步驟路由：分類模型優先，信心不足則 fallback 到 FOLDER_RULES ──
            steps = set()
            if classify_model:
                steps = _classify_folder_steps(root, classify_model,
                                               CLASSIFY_CONF_THRESH, CLASSIFY_CLASS_STEPS) or set()
                if steps:
                    print(f"  [路由-分類模型] 步驟: {sorted(steps)}")
                else:
                    print(f"  [路由-分類模型] 信心不足，回退到 FOLDER_RULES")

            if not steps:  # 分類未啟用 / 信心不足 → 關鍵字比對
                match_target = folder_name + "|" + root
                for keyword, kw_steps in FOLDER_RULES.items():
                    if keyword in match_target:
                        steps.update(kw_steps)
                if not steps:
                    steps = set(DEFAULT_STEPS)
                if steps:
                    print(f"  [路由-關鍵字] 步驟: {sorted(steps)}")

            # Step 2a: YOLO 尺規辨識
            if "yolo" in steps and RULER_YOLO_ENABLED:
                run_yolo(output_json, root, YOLO_MODEL_PATH)

            # Step 2a-alt: A+C 尺規辨識（不需模型）
            if "ruler_ac" in steps:
                run_ruler_ac(output_json, root)

            # Step 2b: 座標 OCR
            if "coord_ocr" in steps:
                run_coord_ocr(output_json, root, COORD_OCR_GROUP_ID)

            # Step 2c: 水準點 OCR
            if "benchmark" in steps:
                run_benchmark_ocr(output_json, root)

    print("\n============ 所有年度處理完成 ============")


if __name__ == "__main__":
    main()
