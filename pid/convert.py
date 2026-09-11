"""DWG(AC1032) → DXF 변환. ODA File Converter를 xvfb-run으로 헤드리스 호출한다."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


def check_tools() -> None:
    """변환에 필요한 외부 도구가 있는지 확인하고, 없으면 설치 명령을 알려준다."""
    missing = []
    if not config.ODA_BIN.exists():
        missing.append(f"ODA File Converter ({config.ODA_BIN}) — `yay -S oda-file-converter`")
    if shutil.which("xvfb-run") is None:
        missing.append("xvfb-run — `sudo pacman -S xorg-server-xvfb`")
    if missing:
        raise RuntimeError("변환 도구 없음:\n  - " + "\n  - ".join(missing))


def convert_dir(src_dir: Path, dst_dir: Path, pattern: str = "*.dwg") -> list[Path]:
    """src_dir의 .dwg를 전부 dst_dir에 .dxf로 변환한다 (폴더 단위 변환만 지원하는 도구다).

    ODA CLI 인자: <입력폴더> <출력폴더> <출력버전> <출력타입> <재귀> <audit> [필터]
    """
    check_tools()
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "xvfb-run", "-a", str(config.ODA_BIN),
        str(src_dir), str(dst_dir),
        config.ODA_OUT_VERSION, "DXF",
        "0",  # 재귀 안 함
        "1",  # audit(복구) 수행
        pattern,
    ]
    logger.info("DWG→DXF 변환: %s → %s", src_dir, dst_dir)
    subprocess.run(cmd, check=True, timeout=600)

    out = sorted(dst_dir.glob("*.dxf"))
    if not out:
        raise RuntimeError(f"변환 결과가 없다: {dst_dir} (원본 경로/패턴 확인)")
    logger.info("변환 완료 %d장", len(out))
    return out
