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
import numpy as np
from PIL import Image, ImageOps
import imagehash

# =============================================
# 設定區 — 請依實際情況修改
# =============================================
BASE_DIR = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi"
YEARS    = ["112"]

# ── 工具路徑 ──────────────────────────────────
YOLO_MODEL_PATH    = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\best.pt"
TESSERACT_CMD      = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
COORD_OCR_GROUP_ID = None  # None = 自動偵測 (輪試 1~18)；整數 = 固定群組
BENCHMARK_GPU      = True

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
    "埋深":  ["ruler_ac", "yolo"],
    "水準點": ["benchmark"],
    "測量讀數照片": ["coord_ocr"],
    "測量坐標讀數": ["coord_ocr"],
    "測量拍照-測量照片讀數照片": ["coord_ocr"],
    "測量照片讀數-照片": ["coord_ocr"],
    "坐標測量": ["coord_ocr"],
}
# 無關鍵字命中時的預設步驟（空清單 = 只做去重）
DEFAULT_STEPS = []
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

_NUM = r'(-?\d+\.\d+|-?\d+)'


def _ocr_preprocess_text(image_path, group_id):
    """影像前處理 + Tesseract OCR，回傳識別文字"""
    import cv2
    import pytesseract

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return ""

    h, w = img.shape[:2]
    if w >= h:
        img = cv2.resize(img, (w, int(w * 16 / 9)), interpolation=cv2.INTER_CUBIC)
    elif group_id in [6, 8, 10, 12, 14, 16, 17]:
        img = cv2.resize(img, (w, int(h * 1.6)), interpolation=cv2.INTER_CUBIC)

    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray   = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray   = cv2.bilateralFilter(gray, d=11, sigmaColor=85, sigmaSpace=85)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 15)
    return pytesseract.image_to_string(thresh, config=r'--psm 6', lang='chi_tra+eng')


def _match_coord_by_group(text, gid):
    """依 group_id 套用對應 Regex，回傳 {"N":..., "E":..., "H_Z":...}"""
    p = _NUM
    N, E, H = "N/A", "N/A", "N/A"

    if gid in [1, 11]:       # 地表N 地表E 地表H
        m = re.search(r'地表.*?[NN]\s*[:=\s]*' + p, text, re.IGNORECASE); N = m.group(1) if m else N
        m = re.search(r'地表.*?[EE]\s*[:=\s]*' + p, text, re.IGNORECASE); E = m.group(1) if m else E
        m = re.search(r'地表.*?[HH]\s*[:=\s]*' + p, text, re.IGNORECASE); H = m.group(1) if m else H

    elif gid in [2, 3, 5]:   # N E Z
        m = re.search(r'[NN][:=\s]*' + p, text, re.IGNORECASE); N = m.group(1) if m else N
        m = re.search(r'[EE][:=\s]*' + p, text, re.IGNORECASE); E = m.group(1) if m else E
        m = re.search(r'[ZZ][:=\s]*' + p, text, re.IGNORECASE); H = m.group(1) if m else H

    elif gid == 4:            # 本地N 本地E H
        m = re.search(r'本地.*?[NN][:=\s]*' + p, text, re.IGNORECASE); N = m.group(1) if m else N
        m = re.search(r'本地.*?[EE][:=\s]*' + p, text, re.IGNORECASE); E = m.group(1) if m else E
        m = re.search(r'(?<!地表)[HH][:=\s]*' + p, text, re.IGNORECASE); H = m.group(1) if m else H

    elif gid in [6, 13]:      # 北 東 高程
        m = re.search(r'北[:=\s]*' + p, text);    N = m.group(1) if m else N
        m = re.search(r'東[:=\s]*' + p, text);    E = m.group(1) if m else E
        m = re.search(r'高程?[:=\s]*' + p, text); H = m.group(1) if m else H

    elif gid == 7:            # NEZ,N,E,Z
        m = re.search(r'NEZ[:=\s]*' + p + r',' + p + r',' + p, text.replace(' ', ''))
        if m: N, E, H = m.group(1), m.group(2), m.group(3)

    elif gid == 8:            # 地表N E H 或 本地N E H
        sn = re.search(r'地表.*?[NN][:=\s]*' + p, text, re.IGNORECASE)
        se = re.search(r'地表.*?[EE][:=\s]*' + p, text, re.IGNORECASE)
        sh = re.search(r'地表.*?[HH][:=\s]*' + p, text, re.IGNORECASE)
        if sn and se and sh:
            N, E, H = sn.group(1), se.group(1), sh.group(1)
        else:
            m = re.search(r'本地.*?[NN][:=\s]*' + p, text, re.IGNORECASE); N = m.group(1) if m else N
            m = re.search(r'本地.*?[EE][:=\s]*' + p, text, re.IGNORECASE); E = m.group(1) if m else E
            m = re.search(r'本地.*?[HH][:=\s]*' + p, text, re.IGNORECASE); H = m.group(1) if m else H

    elif gid == 9:            # N E H
        m = re.search(r'[NN][:=\s]*' + p, text, re.IGNORECASE); N = m.group(1) if m else N
        m = re.search(r'[EE][:=\s]*' + p, text, re.IGNORECASE); E = m.group(1) if m else E
        m = re.search(r'[HH][:=\s]*' + p, text, re.IGNORECASE); H = m.group(1) if m else H

    elif gid in [10, 12, 16, 17]:  # 北座標 東座標 高程
        m = re.search(r'北座標[:=\s]*' + p, text); N = m.group(1) if m else N
        m = re.search(r'東座標[:=\s]*' + p, text); E = m.group(1) if m else E
        m = re.search(r'高程[:=\s]*'   + p, text); H = m.group(1) if m else H

    elif gid == 14:           # 縱軸 橫軸 高程 或 N E Z
        y = re.search(r'縱軸[:=\s]*' + p, text)
        x = re.search(r'橫軸[:=\s]*' + p, text)
        h = re.search(r'高程[:=\s]*'  + p, text)
        if y and x and h:
            N, E, H = y.group(1), x.group(1), h.group(1)
        else:
            m = re.search(r'[NN][:=\s]*' + p, text, re.IGNORECASE); N = m.group(1) if m else N
            m = re.search(r'[EE][:=\s]*' + p, text, re.IGNORECASE); E = m.group(1) if m else E
            m = re.search(r'[ZZ][:=\s]*' + p, text, re.IGNORECASE); H = m.group(1) if m else H

    elif gid == 15:           # 北 東 高度
        m = re.search(r'北[:=\s]*'  + p, text); N = m.group(1) if m else N
        m = re.search(r'東[:=\s]*'  + p, text); E = m.group(1) if m else E
        m = re.search(r'高度[:=\s]*'+ p, text); H = m.group(1) if m else H

    elif gid == 18:           # 本地N 本地E 高程
        m = re.search(r'本地.*?[NN][:=\s]*' + p, text, re.IGNORECASE); N = m.group(1) if m else N
        m = re.search(r'本地.*?[EE][:=\s]*' + p, text, re.IGNORECASE); E = m.group(1) if m else E
        m = re.search(r'高程[:=\s]*'         + p, text);                H = m.group(1) if m else H

    return {"N": N, "E": E, "H_Z": H}


def _fallback_coord(text):
    """通用補救：本地N E 高程，或裸 N E H"""
    p = _NUM
    N, E, H = "N/A", "N/A", "N/A"
    m = re.search(r'本地.*?[NN][:=\s]*' + p, text, re.IGNORECASE); N = m.group(1) if m else N
    m = re.search(r'本地.*?[EE][:=\s]*' + p, text, re.IGNORECASE); E = m.group(1) if m else E
    m = re.search(r'高程[:=\s]*'         + p, text);                H = m.group(1) if m else H
    if N == "N/A":
        m = re.search(r'[NN][:=\s]*' + p, text, re.IGNORECASE); N = m.group(1) if m else N
    if E == "N/A":
        m = re.search(r'[EE][:=\s]*' + p, text, re.IGNORECASE); E = m.group(1) if m else E
    if H == "N/A":
        m = re.search(r'[HH][:=\s]*' + p, text, re.IGNORECASE); H = m.group(1) if m else H
    return {"N": N, "E": E, "H_Z": H}


def _extract_coord(image_path, group_id):
    """辨識單張圖片座標。group_id=None 自動輪試 1~18；整數則固定群組。"""
    if group_id is None:
        best = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}
        for gid in range(1, 19):
            try:
                text = _ocr_preprocess_text(image_path, gid)
                res  = _match_coord_by_group(text, gid)
                if "N/A" not in res.values():
                    return res
                if any(v != "N/A" for v in res.values()):
                    best = res
            except Exception:
                continue
        return best
    else:
        text = _ocr_preprocess_text(image_path, group_id)
        res  = _match_coord_by_group(text, group_id)
        if "N/A" in res.values():
            fb = _fallback_coord(text)
            for k in res:
                if res[k] == "N/A":
                    res[k] = fb[k]
        return res


def run_coord_ocr(output_json_path, folder_path, group_id=None):
    """
    增量座標 OCR。
    只處理 output.json 中尚未有結果的照片。
    結果附加至 <output_dir>/coord_ocr_results.json。
    """
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

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

    print(f"  [座標OCR] 開始辨識 {len(new_items)} 張新照片...")
    for idx, item in enumerate(new_items, 1):
        fn = item["images"]
        fp = os.path.join(folder_path, fn)
        try:
            existing[fn] = _extract_coord(fp, group_id)
        except Exception as e:
            existing[fn] = {"N": "N/A", "E": "N/A", "H_Z": "N/A", "error": str(e)}
        if idx % 50 == 0 or idx == len(new_items):
            print(f"  [座標OCR] 進度: {idx}/{len(new_items)}")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=4)
    print(f"  [座標OCR] 完成！結果儲存至: {results_path}")


# ─────────────────────────────────────────────────────────────
# STEP 2c  水準點 OCR (onelevelwater 邏輯)
# ─────────────────────────────────────────────────────────────

# def run_benchmark_ocr(output_json_path, folder_path):
#     """
#     增量水準點 OCR。
#     只處理 output.json 中尚未有結果的照片。
#     結果附加至 <output_dir>/benchmark_results.json。
#     """
#     import os
#     import json
#     import numpy as np
#     from PIL import Image, ImageOps
#     import easyocr

#     out_dir      = os.path.dirname(output_json_path)
#     results_path = os.path.join(out_dir, "benchmark_results.json")

#     existing = {}
#     if os.path.exists(results_path):
#         with open(results_path, 'r', encoding='utf-8') as f:
#             existing = json.load(f)

#     with open(output_json_path, 'r', encoding='utf-8') as f:
#         new_items = [it for it in json.load(f) if it.get("images") and it["images"] not in existing]

#     if not new_items:
#         print("  [水準點OCR] 無新照片需處理。")
#         return

#     print("  [水準點OCR] 初始化 EasyOCR...")
#     reader = easyocr.Reader(['en'], gpu=BENCHMARK_GPU)
#     print(f"  [水準點OCR] 開始辨識 {len(new_items)} 張新照片...")

#     for idx, item in enumerate(new_items, 1):
#         fn = item["images"]
#         fp = os.path.join(folder_path, fn)
#         try:
#             with Image.open(fp) as img:
#                 img = ImageOps.exif_transpose(img)
#                 rgb = img.convert('RGB')
            
#             final = ""
#             # 遍歷 4 個角度尋找正確的 4 位數
#             for angle in [0, 90, 180, 270]:
#                 cur  = rgb if angle == 0 else rgb.transpose(getattr(Image, f"ROTATE_{angle}"))
#                 hits = reader.readtext(np.array(cur), detail=0, allowlist='0123456789')
                
#                 # 嚴格篩選：只留長度剛好等於 4 的數字
#                 fours = [t for t in hits if len(t) == 4]
#                 if fours:
#                     final = fours[0]
#                     break  # 找到符合的 4 位數就跳出角度迴圈
            
#             # ── 關鍵修改處 ──
#             # 如果跑完 4 個角度依然沒有找到任何剛好 4 位數的結果，就寫入「請重新拍攝」
#             if final:
#                 existing[fn] = final
#             else:
#                 existing[fn] = "請重新拍攝"

#         except Exception:
#             existing[fn] = "Error"

#         if idx % 50 == 0 or idx == len(new_items):
#             print(f"  [水準點OCR] 進度: {idx}/{len(new_items)}")

#     with open(results_path, 'w', encoding='utf-8') as f:
#         json.dump(existing, f, ensure_ascii=False, indent=4)
#     print(f"  [水準點OCR] 完成！結果儲存至: {results_path}")

def run_benchmark_ocr(output_json_path, folder_path):
    """
    增量水準點 OCR。
    結合全圖與中央局部放大（Crop）機制，大幅提升遠距離照片的辨識率。
    """
    import os
    import json
    import numpy as np
    import cv2  # 引入 OpenCV 處理影像縮放效果較佳
    from PIL import Image, ImageOps
    import easyocr

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

    print("  [水準點OCR] 初始化 EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True) # 建議開啟 GPU 加速
    print(f"  [水準點OCR] 開始辨識 {len(new_items)} 張新照片...")

    for idx, item in enumerate(new_items, 1):
        fn = item["images"]
        fp = os.path.join(folder_path, fn)
        try:
            with Image.open(fp) as img:
                img = ImageOps.exif_transpose(img)
                rgb = img.convert('RGB')
            
            final = ""
            
            # === 策略一：先用原圖跑 4 個角度 ===
            for angle in [0, 90, 180, 270]:
                cur = rgb if angle == 0 else rgb.transpose(getattr(Image, f"ROTATE_{angle}"))
                hits = reader.readtext(np.array(cur), detail=0, allowlist='0123456789')
                fours = [t for t in hits if len(t) == 4]
                if fours:
                    final = fours[0]
                    break
            
            # === 策略二：如果原圖失敗，切出中央區域（模擬人工放大） ===
            if not final:
                w, h = rgb.size
                # 定義中央區域（例如取中央 30% 區塊，可依照片習慣調整）
                crop_w, crop_h = int(w * 0.3), int(h * 0.3)
                left = (w - crop_w) // 2
                top = (h - crop_h) // 2
                right = left + crop_w
                bottom = top + crop_h
                
                # 裁剪並利用雙立方插值放大 2 倍，增強文字邊緣
                cropped = rgb.crop((left, top, right, bottom))
                cropped_resized = cropped.resize((crop_w * 2, crop_h * 2), Image.Resampling.BICUBIC)
                
                # 對裁剪後的局部圖再次進行 4 角度辨識
                for angle in [0, 90, 180, 270]:
                    cur = cropped_resized if angle == 0 else cropped_resized.transpose(getattr(Image, f"ROTATE_{angle}"))
                    hits = reader.readtext(np.array(cur), detail=0, allowlist='0123456789')
                    fours = [t for t in hits if len(t) == 4]
                    if fours:
                        final = fours[0]
                        break

            # ── 結果寫入 ──
            if final:
                existing[fn] = final
            else:
                existing[fn] = "請重新拍攝"

        except Exception as e:
            existing[fn] = f"Error: {str(e)}"

        if idx % 50 == 0 or idx == len(new_items):
            print(f"  [水準點OCR] 進度: {idx}/{len(new_items)}")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=4)
    print(f"  [水準點OCR] 完成！結果儲存至: {results_path}")
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

        for root, dirs, _ in os.walk(year_dir):
            dirs[:] = [d for d in dirs if not d.endswith("_output")]
            folder_name = os.path.basename(root)
            if folder_name == year:
                folder_name = f"{year}年度"

            # Step 1: 去重，產生 output.json
            output_json = dedup_folder(root, folder_name, year)
            if not output_json:
                continue

            # 依資料夾名稱關鍵字決定步驟（比對 folder_name 與完整路徑）
            match_target = folder_name + "|" + root
            steps = set()
            for keyword, kw_steps in FOLDER_RULES.items():
                if keyword in match_target:
                    steps.update(kw_steps)
            if not steps:
                steps = set(DEFAULT_STEPS)

            if steps:
                print(f"  [路由] 步驟: {sorted(steps)}")

            # Step 2a: YOLO 尺規辨識
            if "yolo" in steps:
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
