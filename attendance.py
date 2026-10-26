import time
from collections import defaultdict
import cv2
import db
from config import (CAMERA_INDEX, MATCH_THRESHOLD, MARGIN, CONFIRM_FRAMES, PROCESS_EVERY_N)
from face_engine import load_engine, detect_faces, match


def main():
    db.init_db()
    gallery, gids, students = db.load_gallery()
    if len(gids) == 0:
        print("هیچ دانش‌آموزی ثبت‌نام نشده. اول enroll.py را اجرا کنید.")
        return
    print(f"{len(students)} دانش‌آموز بارگذاری شد. در حال بارگذاری مدل...")
    app = load_engine()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("دوربین باز نشد.")
        return

    marked = db.present_ids_today()
    streak = defaultdict(int)
    results = []          # آخرین نتایج برای رسم روی تصویر
    banner, banner_until = "", 0
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1

        if frame_no % PROCESS_EVERY_N == 0:
            results = []
            seen = set()
            for f in detect_faces(app, frame):
                sid, score = match(f.normed_embedding, gallery, gids, MATCH_THRESHOLD, MARGIN)
                results.append((f.bbox, sid, score))
                if sid is not None:
                    seen.add(sid)
                    streak[sid] += 1
                    if streak[sid] >= CONFIRM_FRAMES and sid not in marked:
                        if db.mark_present(sid, score):
                            code, name = students[sid]
                            print(f"✔ حضور ثبت شد: {name} ({code}) - شباهت {score:.2f}")
                            banner, banner_until = f"PRESENT: {code}", time.time() + 2.0
                        marked.add(sid)
            for sid in list(streak):        
                if sid not in seen:
                    del streak[sid]

        for bbox, sid, score in results:
            x1, y1, x2, y2 = map(int, bbox)
            if sid is None:
                color, label = (0, 0, 255), "Unknown"
            elif sid in marked:
                color, label = (0, 200, 0), f"{students[sid][0]} OK"
            else:
                color, label = (0, 200, 255), f"{students[sid][0]} ..."
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Present: {len(marked)}/{len(students)}   [Q]=quit",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if time.time() < banner_until:
            cv2.putText(frame, banner, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"پایان. حاضر: {len(marked)} از {len(students)}. برای دیدن غایبین: python report.py")


if __name__ == "__main__":
    main()
