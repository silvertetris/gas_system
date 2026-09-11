"""P&ID 판독 파이프라인 진입점.

    .venv/bin/python -m pid.pipeline

DWG → DXF 변환 → 텍스트 JSON 추출 → 도면별 전체 PNG 렌더까지 수행한다.
확대 판독은 pid.extract.render(dxf, png, box=(x0,y0,x1,y1))로 개별 호출한다.
판독 결과는 docs/htr31p_flow.md §7 참조.
"""
from __future__ import annotations

import logging

from . import config, convert, extract

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pid.pipeline")


def main() -> None:
    dxf_files = convert.convert_dir(config.PID_DWG_DIR, config.DXF_DIR)
    extract.extract_all(config.DXF_DIR, config.TEXT_JSON)
    for dxf in dxf_files:
        png = config.PNG_DIR / (dxf.stem + ".png")
        extract.render(dxf, png)
        logger.info("렌더 완료 %s", png.name)
    logger.info("산출물: %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
