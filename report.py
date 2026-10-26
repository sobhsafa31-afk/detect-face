"""گزارش حضور و غیاب
python report.py                  -> گزارش امروز
python report.py --csv            -> خروجی اکسل (CSV)
python report.py --present 1234   -> ثبت دستی حضور (تأیید معلم)
python report.py --day 2025-01-31 -> گزارش یک روز دیگر
"""
import argparse
import csv
import db


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="store_true")
    p.add_argument("--present", metavar="CODE")
    p.add_argument("--day")
    a = p.parse_args()
    db.init_db()

    if a.present:
        with db.connect() as c:
            row = c.execute("SELECT id, name FROM students WHERE code=?", (a.present,)).fetchone()
        if not row:
            print("چنین کدی پیدا نشد.")
        elif db.mark_present(row[0], None, source="manual"):
            print(f"✔ حضور {row[1]} به صورت دستی ثبت شد.")
        else:
            print("این دانش‌آموز امروز قبلاً ثبت شده بود.")
        return

    present, absent = db.report(a.day)
    day = a.day or db.today()
    print(f"\n===== گزارش {day} =====")
    print(f"حاضر: {len(present)}   غایب (تأیید‌نشده): {len(absent)}\n")
    print("--- غایبین ---")
    for code, name in absent:
        print(f"{code}\t{name}")

    if a.csv:
        fn = f"attendance_{day}.csv"
        with open(fn, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig برای فارسی در اکسل
            w = csv.writer(f)
            w.writerow(["کد", "نام", "وضعیت", "ساعت", "روش"])
            for code, name, t, score, src in present:
                w.writerow([code, name, "حاضر", t, src])
            for code, name in absent:
                w.writerow([code, name, "غایب", "", ""])
        print(f"\nفایل {fn} ساخته شد.")


if __name__ == "__main__":
    main()
