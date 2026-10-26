import cv2
import numpy as np
import db
from config import CAMERA_INDEX, ENROLL_PHOTOS, MATCH_THRESHOLD
from face_engine import load_engine, detect_faces


def main():
    db.init_db()
    code = input("شماره/کد دانش‌آموزی: ").strip()
    name = input("نام و نام خانوادگی: ").strip()
    if not code or not name:
        print("کد و نام نباید خالی باشد.")
        return

    print("در حال بارگذاری مدل (بار اول چند دقیقه طول می‌کشد چون مدل دانلود می‌شود)...")
    app = load_engine()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("دوربین باز نشد. CAMERA_INDEX را در config.py تغییر دهید.")
        return

    captured = []
    tips = ["Look straight", "Turn head slightly LEFT", "Turn head slightly RIGHT",
            "Chin slightly UP", "Chin slightly DOWN"]
    while len(captured) < ENROLL_PHOTOS:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detect_faces(app, frame)
        view = frame.copy()
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0) if len(faces) == 1 else (0, 0, 255), 2)
        tip = tips[len(captured) % len(tips)]
        cv2.putText(view, f"{len(captured)}/{ENROLL_PHOTOS}  {tip}  [SPACE]=capture [Q]=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Enroll", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("انصراف داده شد.")
            break
        if key == 32:
            if len(faces) != 1:
                print("دقیقاً یک چهره باید جلوی دوربین باشد (و به اندازهٔ کافی نزدیک).")
                continue
            captured.append(faces[0].normed_embedding)
            print(f"عکس {len(captured)} از {ENROLL_PHOTOS} ثبت شد.")

    cap.release()
    cv2.destroyAllWindows()
    if len(captured) < ENROLL_PHOTOS:
        print("ثبت‌نام کامل نشد و ذخیره نشد.")
        return

    # هشدار اگر این چهره خیلی شبیه دانش‌آموز دیگری باشد
    mat, ids, students = db.load_gallery()
    sid, _ = db.get_or_create_student(code, name)
    if len(ids):
        others = ids != sid
        if others.any():
            sims = mat[others] @ np.mean(captured, axis=0)
            j = int(np.argmax(sims))
            if sims[j] >= MATCH_THRESHOLD:
                other = students[int(ids[others][j])]
                print(f"⚠ هشدار: این چهره شبیه {other[1]} (کد {other[0]}) است. مطمئن شوید اشتباه نشده.")

    db.add_embeddings(sid, captured)
    print(f"✔ {name} با کد {code} ثبت شد.")


if __name__ == "__main__":
    main()
