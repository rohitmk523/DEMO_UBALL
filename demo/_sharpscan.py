"""Throwaway: sharpest DEINTERLACED frame per camera (real blur metric)."""
import subprocess, sys, tempfile, concurrent.futures as cf
from pathlib import Path
import cv2
BASE=("s3://uball-videos-production/court-a/2026-03-19/"
      "c2a354fe-eb34-4980-af00/2026-03-19_c2a354fe-eb34-4980-af00")
ANGLE=sys.argv[1] if len(sys.argv)>1 else "FL"
N=int(sys.argv[2]) if len(sys.argv)>2 else 60
url=subprocess.run(["aws","s3","presign",f"{BASE}_{ANGLE}.mp4","--expires-in","2400"],
                   check=True,capture_output=True,text=True).stdout.strip()
times=[60+int(i*(3597-120)/(N-1)) for i in range(N)]
def score(t):
    f=Path(tempfile.gettempdir())/f"_ds_{ANGLE}_{t}.jpg"
    try:
        subprocess.run(["ffmpeg","-y","-loglevel","error","-ss",str(t),"-i",url,
                        "-vf","field=top,scale=1920:1080","-frames:v","1","-q:v","2",str(f)],
                       check=True,timeout=60)
        img=cv2.imread(str(f))
        if img is None: return (t,-1.0)
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return (t,float(cv2.Laplacian(g,cv2.CV_64F).var()))
    except Exception: return (t,-1.0)
    finally: f.unlink(missing_ok=True)
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    res=sorted(ex.map(score,times),key=lambda r:-r[1])
print(f"=== {ANGLE} DEINTERLACED sharpness (higher=sharper) ===")
for t,s in res[:15]: print(f"  t={t:5d}  lapvar={s:8.1f}")
print("TOP:",[t for t,_ in res[:8]])
