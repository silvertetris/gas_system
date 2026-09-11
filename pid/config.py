"""P&ID 도면 판독 설정.

영종 G/S 도면 16장 중 배관 위상(topology)을 담은 P&ID 6장이 대상이다.
원본은 AC1032(AutoCAD 2018) 바이너리라 파이썬만으로는 못 연다 —
ODA File Converter로 DXF로 바꾼 뒤 ezdxf로 읽는다 (§설치 요구사항).

설치 요구사항 (시스템 패키지, 최초 1회):
    yay -S oda-file-converter        # AUR. /opt/oda-file-converter/oda-file-converter
    sudo pacman -S xorg-server-xvfb  # GUI 앱이라 헤드리스 실행에 xvfb-run 필요
    .venv/bin/pip install ezdxf      # requirements.txt에 포함
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 원본 .dwg 위치 (16장; 이 중 P&ID 6장이 P&ID/ 하위에 있다)
DWG_ROOT = DATA_DIR / "영종GS 현장 세부데이터" / "260410 영종GS 도면"
PID_DWG_DIR = DWG_ROOT / "P&ID"

# 변환·판독 산출물 (data/ 이하라 git에 커밋되지 않는다 — .gitignore)
OUTPUT_DIR = DWG_ROOT / "_converted"
DXF_DIR = OUTPUT_DIR / "dxf"
PNG_DIR = OUTPUT_DIR / "png"
TEXT_JSON = OUTPUT_DIR / "pid_text.json"

# ODA File Converter 실행파일. AUR 패키지가 PATH에 심볼릭 링크를 만들지 않으므로 절대경로를 쓴다.
ODA_BIN = Path("/opt/oda-file-converter/oda-file-converter")
ODA_OUT_VERSION = "ACAD2018"  # 원본과 동일 버전으로 내보낸다 (다운그레이드 시 엔티티 손실 위험)

# 도면 번호 → 내용 (판독으로 확인, 2026-09-11)
SHEETS = {
    "09-15-R-32-001": "히터 계통 — 필터 F-21A/B/O/P, HTR-31O, HTR-31P, 합류헤더 31Z",
    "09-15-R-32-002": "정압기 O/P/Q/R — PCV-41O/P/Q/R + 42x, 계량 FU-61O/P/Q",
    "09-15-R-32-003": "히터 내부 계통 (HTR-31O 상세, 연료가스)",
    "09-15-R-32-004": "히터 A/B 계통 — HTR-31A, HTR-31B (인천도시가스 계열)",
    "09-15-R-32-005": "정압기 A/B/C/D — PCV-41A/B/C, 41D/42D",
    "09-15-R-32-006": "히터 내부 계통 (상세, 수조·버너)",
}

# 렌더 기본값
RENDER_FIGSIZE = (20, 14)
RENDER_DPI = 200
