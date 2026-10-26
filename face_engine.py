import numpy as np
from insightface.app import FaceAnalysis
from config import MODEL_NAME, DET_SIZE, MIN_DET_SCORE, MIN_FACE_SIZE, MAX_FACES


def load_engine():
    # اگر کارت گرافیک NVIDIA دارید و onnxruntime-gpu نصب است، CUDAExecutionProvider را اول بگذارید
    app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    return app


def face_width(face):
    x1, y1, x2, y2 = face.bbox
    return min(x2 - x1, y2 - y1)


def detect_faces(app, frame):
    """فقط چهره‌های با کیفیت کافی (به ترتیب بزرگ‌ترین) را برمی‌گرداند"""
    faces = [f for f in app.get(frame)
             if f.det_score >= MIN_DET_SCORE and face_width(f) >= MIN_FACE_SIZE]
    faces.sort(key=face_width, reverse=True)
    return faces[:MAX_FACES]


def match(embedding, gallery_mat, gallery_ids, threshold, margin):
    """شبیه‌ترین دانش‌آموز را پیدا می‌کند. اگر مطمئن نبود None برمی‌گرداند."""
    if len(gallery_ids) == 0:
        return None, 0.0
    sims = gallery_mat @ embedding            # بردارها نرمال هستند => ضرب داخلی = شباهت کسینوسی
    uniq = np.unique(gallery_ids)
    best = np.full(len(uniq), -1.0)
    for i, sid in enumerate(uniq):
        best[i] = sims[gallery_ids == sid].max()
    order = np.argsort(best)[::-1]
    top = best[order[0]]
    second = best[order[1]] if len(order) > 1 else -1.0
    if top >= threshold and (top - second) >= margin:
        return int(uniq[order[0]]), float(top)
    return None, float(top)
