# -*- coding: utf-8 -*-
from ultralytics import YOLO

paths = {
    "classify": r"D:\work2\0526ocr\影像分類best.pt",
    "ruler":    r"D:\work2\0526ocr\尺規bestv3.pt",
    "coordfmt": r"D:\work2\0526ocr\判斷格式分類best2.pt",
    "bench":    r"D:\work2\0526ocr\一等水準點best.pt",
}
for tag, p in paths.items():
    try:
        m = YOLO(p)
        print(f"[{tag}] task={m.task}  names={m.names}")
    except Exception as e:
        print(f"[{tag}] ERROR: {e}")
