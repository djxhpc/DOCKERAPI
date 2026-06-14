# -*- coding: utf-8 -*-
"""
auto_classify_pipeline.py — 逐張影像自動分類 + 辨識 + 去重輸出

用途:
  同事把影像丟進「收件匣」資料夾，本程式會「逐張」自動：
    1. 去重     : SHA-256 精確比對（完全相同）＋ pHash16 疑似重複偵測，重複也照樣寫進 JSON。
    2. 分類     : 用分類模型（.pt）判斷這張是哪種影像 → 走對應判斷邏輯（埋深/座標/水準點）。
    3. 未知退路 : 分不出類型 / 分類信心不足 / 分類了但擠不出值時，
                  把三種判斷邏輯全部試一遍，哪個有有效輸出就採用（可用開關關閉）。
    4. 輸出     : 全部結果整合進單一 auto_results.json（以檔名為 key），可直接回傳給同事。

兩種執行模式:
  一次性批次:  python auto_classify_pipeline.py --input <收件匣> --output <輸出夾>
  常駐監控  :  python auto_classify_pipeline.py --watch  --input <收件匣> --output <輸出夾>
              （每隔幾秒掃描收件匣，偵測到新影像就自動分析，Ctrl+C 停止）

設計理念:
  本檔「不重複」實作辨識邏輯，而是 import pipeline0611 直接重用其底層函式
  （座標 regex、TWD97 驗證、尺規交叉判斷、水準點裁切…），原邏輯更新本程式自動跟上。
"""
import os
import json
import time
import argparse
import unicodedata
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps
import imagehash
import cv2

import pipeline0611 as pl   # 重用既有辨識邏輯


# =============================================
# 設定區 — 請依實際情況修改
# =============================================
# ── 收件匣 / 輸出夾 ────────────────────────────
INPUT_DIR  = r"D:\work2\0526ocr\收件匣"          # 同事丟影像的資料夾
OUTPUT_DIR = r"D:\work2\0526ocr\收件匣_輸出"     # auto_results.json 等輸出位置

# ── 模型路徑（沿用 pipeline0611 的設定，可在此覆寫）──
CLASSIFY_MODEL_PATH = pl.CLASSIFY_MODEL_PATH     # 逐張影像分類模型 best.pt
RULER_MODEL_PATH    = pl.YOLO_MODEL_PATH         # 尺規（埋深）YOLO 模型

# ── 分類類別名稱 → 判斷邏輯類型 ────────────────
# key 必須與分類模型訓練時的類別名稱一致；value 為 None 代表「不辨識」。
# 類型代號: "ruler"(尺規/埋深) | "coord"(座標OCR) | "benchmark"(水準點OCR)
CLASS_TO_TYPE = {
    "埋深":   "ruler",
    "水準點": "benchmark",
    "座標":   "coord",
    "其他":   None,
}

# 分類信心低於此值 → 視為「未知」，走 try-all 退路
CLASSIFY_CONF_THRESH = 0.6

# 【未知退路自動開關】
#   True  : 分不出類型 / 信心不足 / 分類了卻擠不出值時，三種邏輯全試一遍取有效者
#   False : 直接標記 unknown，不做 OCR 比對退路
ENABLE_UNKNOWN_FALLBACK = True

# try-all 退路的嘗試優先序（會輸出實際數值的排前面，較容易判定有效）
FALLBACK_PRIORITY = ["coord", "benchmark", "ruler"]

# 監控模式掃描間隔（秒）
WATCH_INTERVAL = 5
# =============================================


# ─────────────────────────────────────────────────────────────
# 小工具
# ─────────────────────────────────────────────────────────────
def _load_bgr(path):
    """讀檔 → 修正 EXIF 方向 → 轉 BGR ndarray（給 YOLO 用）"""
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _phash16(path):
    try:
        return str(imagehash.phash(Image.open(path), hash_size=16))
    except Exception:
        return None


def _pick_four(result):
    """從 RapidOCR result 取出信心值 > 0.4 的第一個 4 位純數字，回傳 (text, conf)"""
    if not result:
        return "", 0.0
    for item in result:
        text, conf = item[1], float(item[2])
        if conf > 0.4 and len(text) == 4 and text.isdigit():
            return text, round(conf, 4)
    return "", 0.0


# ─────────────────────────────────────────────────────────────
# 三種判斷邏輯（逐張版，重用 pipeline0611 底層函式）
# ─────────────────────────────────────────────────────────────
def analyze_ruler(img_bgr, ruler_model):
    """尺規（埋深）：YOLO 偵測 vertical / horizontal → 交叉判斷。"""
    preds = ruler_model.predict(
        source=img_bgr, imgsz=1280,
        conf=min(pl.HORIZONTAL_CONF_THRESHOLD, pl.VERTICAL_CONF_THRESHOLD),
        iou=0.5, augment=True, verbose=False,
    )[0]

    vertical, horizontal = [], []
    for box in (preds.boxes or []):
        bbox = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        name = pl._get_class_name(ruler_model, box.cls[0])
        if name == "vertical" and conf >= pl.VERTICAL_CONF_THRESHOLD:
            vertical.append(bbox)
        elif name == "horizontal" and conf >= pl.HORIZONTAL_CONF_THRESHOLD:
            horizontal.append(bbox)

    status, is_crossed, best_iou = pl._classify_crossing(vertical, horizontal)
    return {
        "status": status,
        "vertical_count": len(vertical),
        "horizontal_count": len(horizontal),
        "is_crossed": is_crossed,
        "best_iou": round(best_iou, 4),
    }


def analyze_coord(path, engine):
    """座標 OCR：全圖辨識 → 通用 pattern 比對 → TWD97 位數 fallback。"""
    text = pl._ocr_with_paddle(path, engine)
    text = unicodedata.normalize("NFKC", text)
    text = pl._strip_dms(text)

    res = pl._match_coord(text, pl._COORD_PATTERNS)
    if not pl._validate_twd97(res):
        fb = pl._digit_fallback_twd97(text)
        if pl._validate_twd97(fb):
            res = fb
        else:
            # 僅補上原本缺漏（N/A）的欄位
            for k in ("N", "E", "H_Z"):
                if res.get(k) == "N/A" and fb.get(k) != "N/A":
                    res[k] = fb[k]

    res["needs_review"] = "正常" if pl._validate_twd97(res) else "請重新拍攝"
    return res


def analyze_benchmark(path, reader, yolo_model):
    """水準點 OCR：YOLO 裁切(選用) → 全圖 → 中央放大，找 4 位數字。"""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    arr = np.array(img)

    number, conf = "", 0.0
    if yolo_model:
        for crop in pl._yolo_crop_regions(arr, yolo_model,
                                          pl.BENCHMARK_YOLO_CONF, pl.BENCHMARK_YOLO_PADDING):
            result, _ = reader(np.array(crop))
            number, conf = _pick_four(result)
            if number:
                break

    if not number:
        result, _ = reader(arr)
        number, conf = _pick_four(result)

    if not number:
        w, h = img.size
        for ratio in (0.3, 0.5):
            cw, ch = int(w * ratio), int(h * ratio)
            left, top = (w - cw) // 2, (h - ch) // 2
            crop = img.crop((left, top, left + cw, top + ch)).resize(
                (cw * 2, ch * 2), Image.Resampling.BICUBIC)
            result, _ = reader(np.array(crop))
            number, conf = _pick_four(result)
            if number:
                break

    if number:
        return {"number": number, "ocr_conf": conf, "needs_review": False}
    return {"number": "請重新拍攝", "ocr_conf": None, "needs_review": True}


def run_branch(typ, path, models, img_bgr=None):
    """依類型分派到對應判斷邏輯。"""
    if typ == "ruler":
        if img_bgr is None:
            img_bgr = _load_bgr(path)
        return analyze_ruler(img_bgr, models["ruler"])
    if typ == "coord":
        return analyze_coord(path, models["ocr"])
    if typ == "benchmark":
        return analyze_benchmark(path, models["ocr"], models["benchmark_yolo"])
    return None


def _safe_run(typ, path, models, img_bgr=None):
    try:
        return run_branch(typ, path, models, img_bgr)
    except Exception as e:
        return {"error": str(e)}


def _is_positive(typ, result):
    """判斷某類型的辨識結果是否「擠得出有效值」。"""
    if not result or "error" in result:
        return False
    if typ == "ruler":
        return result.get("status") == "Normal (Crossed)"
    if typ == "coord":
        return result.get("needs_review") == "正常"
    if typ == "benchmark":
        return result.get("number") not in (None, "", "請重新拍攝", "Error")
    return False


def try_all(path, models):
    """三種邏輯全試一遍，依優先序回傳第一個有效類型。回傳 (chosen_type | None, attempts_dict)。"""
    img_bgr = None
    try:
        img_bgr = _load_bgr(path)
    except Exception:
        pass

    attempts = {}
    for typ in FALLBACK_PRIORITY:
        # 模型沒載入的類型直接跳過
        if typ == "ruler" and not models.get("ruler"):
            continue
        if typ in ("coord", "benchmark") and not models.get("ocr"):
            continue
        attempts[typ] = _safe_run(typ, path, models, img_bgr)

    chosen = None
    for typ in FALLBACK_PRIORITY:
        if _is_positive(typ, attempts.get(typ)):
            chosen = typ
            break
    return chosen, attempts


# ─────────────────────────────────────────────────────────────
# 分類
# ─────────────────────────────────────────────────────────────
def classify_image(path, model):
    """回傳 (class_name | None, conf)。"""
    if model is None:
        return None, 0.0
    try:
        res = model(path)[0]
        return res.names[res.probs.top1], float(res.probs.top1conf)
    except Exception:
        return None, 0.0


def _overall_review(typ, result):
    """整體狀態（給同事看的一句話）。"""
    if typ == "unknown" or result is None:
        return "未知，請人工確認"
    if typ == "ruler":
        return "正常" if result.get("status") == "Normal (Crossed)" else "請重新拍攝"
    if typ == "coord":
        return result.get("needs_review", "請重新拍攝")
    if typ == "benchmark":
        return "請重新拍攝" if result.get("needs_review") else "正常"
    return "未知，請人工確認"


# ─────────────────────────────────────────────────────────────
# 狀態存放（單一 JSON，逐張增量）
# ─────────────────────────────────────────────────────────────
class State:
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.results_path = os.path.join(output_dir, "auto_results.json")
        self.hashes_path  = os.path.join(output_dir, "auto_hashes.json")

        self.results = self._load(self.results_path)   # filename -> record
        self.hashes  = self._load(self.hashes_path)     # filename -> {sha256, phash16}

        # 反查索引
        self.sha_index   = {}   # sha256 -> filename
        self.phash_index = {}   # filename -> phash16
        for fn, h in self.hashes.items():
            if not isinstance(h, dict):
                continue
            if h.get("sha256"):
                self.sha_index.setdefault(h["sha256"], fn)
            if h.get("phash16"):
                self.phash_index[fn] = h["phash16"]

    @staticmethod
    def _load(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
        with open(self.hashes_path, "w", encoding="utf-8") as f:
            json.dump(self.hashes, f, ensure_ascii=False, indent=4)


# ─────────────────────────────────────────────────────────────
# 主處理：逐張
# ─────────────────────────────────────────────────────────────
def process_image(path, models, state):
    fn = os.path.basename(path)
    rec = {
        "filename": fn,
        "absolute_path": os.path.abspath(path),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }

    # ── Step 1a: SHA-256 精確去重 ──────────────────────────────
    sha = pl._get_file_sha256(path)
    if sha and sha in state.sha_index and state.sha_index[sha] != fn:
        orig_fn = state.sha_index[sha]
        orig = state.results.get(orig_fn, {})
        rec.update(
            is_duplicate=True,
            duplicate_of=orig_fn,
            is_suspected_duplicate=False,
            classify=orig.get("classify"),
            type=orig.get("type"),
            result=orig.get("result"),
            needs_review=orig.get("needs_review"),
            note=f"與 {orig_fn} 完全相同（SHA-256），沿用其辨識結果",
        )
        state.results[fn] = rec
        state.hashes[fn] = {"sha256": sha}
        state.save()
        print(f"  [重複] {fn} ← 與 {orig_fn} 完全相同，沿用結果")
        return rec

    rec["is_duplicate"] = False
    rec["duplicate_of"] = None

    # ── Step 1b: pHash16 疑似重複 ──────────────────────────────
    ph = _phash16(path)
    susp = None
    if pl.PHASH_SUSPECTED_ENABLED and ph:
        ph_obj = imagehash.hex_to_hash(ph)
        best_d, best = pl.PHASH_SUSPECTED_THRESHOLD + 1, None
        for ofn, oph in state.phash_index.items():
            if ofn == fn:
                continue
            d = ph_obj - imagehash.hex_to_hash(oph)
            if d <= pl.PHASH_SUSPECTED_THRESHOLD and d < best_d:
                best_d, best = d, ofn
        if best is not None:
            susp = {"similar_to": best, "distance": int(best_d)}
    rec["is_suspected_duplicate"] = susp is not None
    rec["suspected"] = susp

    # 先登記 hash（讓同批次後續影像能比對到這張）
    if sha:
        state.sha_index[sha] = fn
    if ph:
        state.phash_index[fn] = ph
    state.hashes[fn] = {"sha256": sha, "phash16": ph}

    # ── Step 2: 分類 ──────────────────────────────────────────
    cls, conf = classify_image(path, models.get("classify"))
    mapped = CLASS_TO_TYPE.get(cls) if cls else None

    typ, source, result, fallback_attempts = "unknown", "none", None, None

    if cls and conf >= CLASSIFY_CONF_THRESH and mapped:
        # 分類成功 → 走對應邏輯
        typ, source = mapped, "model"
        result = _safe_run(typ, path, models)
        # 分類了卻擠不出值 → 反查其他邏輯能不能輸出值
        if not _is_positive(typ, result) and ENABLE_UNKNOWN_FALLBACK:
            chosen, attempts = try_all(path, models)
            fallback_attempts = attempts
            if chosen:
                typ, result, source = chosen, attempts[chosen], "fallback(原分類無有效輸出)"
            else:
                source = "model(無有效輸出)"
    else:
        # 未知 / 信心不足
        if ENABLE_UNKNOWN_FALLBACK:
            chosen, attempts = try_all(path, models)
            fallback_attempts = attempts
            if chosen:
                typ, result, source = chosen, attempts[chosen], "fallback(未分類)"
            else:
                typ, source = "unknown", "fallback(全部無有效輸出)"
        else:
            typ, source = "unknown", "分類信心不足且退路關閉"

    rec["classify"] = {
        "class": cls,
        "conf": round(conf, 4) if cls else None,
        "threshold": CLASSIFY_CONF_THRESH,
        "source": source,
    }
    rec["type"] = typ
    rec["result"] = result
    if fallback_attempts is not None:
        rec["fallback_attempts"] = fallback_attempts
    rec["needs_review"] = _overall_review(typ, result)

    state.results[fn] = rec
    state.save()

    flag = "（疑似重複）" if rec["is_suspected_duplicate"] else ""
    print(f"  [完成] {fn} → 類型={typ} 來源={source} 狀態={rec['needs_review']} {flag}")
    return rec


# ─────────────────────────────────────────────────────────────
# 批次 / 監控
# ─────────────────────────────────────────────────────────────
def _list_images(input_dir):
    return sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(pl._IMG_EXT) and os.path.isfile(os.path.join(input_dir, f))
    )


def process_folder(input_dir, state, models):
    todo = [f for f in _list_images(input_dir) if f not in state.results]
    if not todo:
        print("  無新影像需處理。")
        return
    print(f"  共 {len(todo)} 張新影像，開始處理...")
    for i, fn in enumerate(todo, 1):
        process_image(os.path.join(input_dir, fn), models, state)
        if i % 50 == 0 or i == len(todo):
            print(f"  進度: {i}/{len(todo)}")
    print(f"  完成！結果 → {state.results_path}")


def watch_folder(input_dir, state, models, interval):
    print(f"監控中：{input_dir}")
    print(f"每 {interval}s 掃描一次，偵測到新影像即自動分析。按 Ctrl+C 停止。")
    try:
        while True:
            todo = [f for f in _list_images(input_dir) if f not in state.results]
            for fn in todo:
                process_image(os.path.join(input_dir, fn), models, state)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n已停止監控。")


# ─────────────────────────────────────────────────────────────
# 模型載入
# ─────────────────────────────────────────────────────────────
def load_models():
    from ultralytics import YOLO as YOLOModel
    models = {"classify": None, "ruler": None, "ocr": None, "benchmark_yolo": None}

    # 分類模型
    try:
        models["classify"] = YOLOModel(CLASSIFY_MODEL_PATH)
        print(f"[初始化] 分類模型載入成功: {CLASSIFY_MODEL_PATH}")
    except Exception as e:
        print(f"[初始化] 分類模型載入失敗: {e}（將一律走未知退路）")

    # 尺規模型
    try:
        models["ruler"] = YOLOModel(RULER_MODEL_PATH)
        print(f"[初始化] 尺規模型載入成功: {RULER_MODEL_PATH}")
    except Exception as e:
        print(f"[初始化] 尺規模型載入失敗: {e}")

    # OCR 引擎（座標 + 水準點共用）
    try:
        from rapidocr_onnxruntime import RapidOCR
        models["ocr"] = RapidOCR()
        print("[初始化] RapidOCR 載入成功")
    except Exception as e:
        print(f"[初始化] RapidOCR 載入失敗: {e}")

    # 水準點裁切模型（選用）
    if pl.BENCHMARK_YOLO_CROP:
        try:
            models["benchmark_yolo"] = YOLOModel(pl.BENCHMARK_YOLO_MODEL_PATH)
            print(f"[初始化] 水準點裁切模型載入成功: {pl.BENCHMARK_YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[初始化] 水準點裁切模型載入失敗: {e}")

    return models


def main():
    p = argparse.ArgumentParser(description="逐張影像自動分類 + 辨識 + 去重")
    p.add_argument("--input",  default=INPUT_DIR,  help="收件匣資料夾")
    p.add_argument("--output", default=OUTPUT_DIR, help="輸出資料夾")
    p.add_argument("--watch",  action="store_true", help="常駐監控模式")
    p.add_argument("--interval", type=int, default=WATCH_INTERVAL, help="監控掃描間隔(秒)")
    args = p.parse_args()

    if not os.path.isdir(args.input):
        print(f"[錯誤] 找不到收件匣: {args.input}")
        return

    models = load_models()
    state = State(args.output)

    if args.watch:
        watch_folder(args.input, state, models, args.interval)
    else:
        print(f"\n{'='*10} 批次處理 {'='*10}\n  來源: {args.input}")
        process_folder(args.input, state, models)
    print("\n============ 結束 ============")


if __name__ == "__main__":
    main()
