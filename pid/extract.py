"""DXF에서 판독용 정보를 뽑는다 — 텍스트(좌표 포함) 추출과 PNG 렌더."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import ezdxf
import matplotlib
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing import config as draw_config
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

from . import config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (Agg 백엔드 지정 후 import해야 한다)

logger = logging.getLogger(__name__)


def extract_text(dxf_path: Path) -> list[dict]:
    """TEXT / MTEXT / INSERT의 ATTRIB을 좌표와 함께 뽑는다.

    P&ID의 계기 태그(`TI-21Z` 등)는 대부분 블록 참조의 ATTRIB으로 들어 있어서
    TEXT만 읽으면 절반을 놓친다. 반환 키: t(내용) x y k(종류) l(레이어).
    """
    doc = ezdxf.readfile(dxf_path)
    items: list[dict] = []

    def add(txt: str | None, x: float, y: float, kind: str, layer: str) -> None:
        txt = (txt or "").strip()
        if txt:
            items.append({"t": txt, "x": round(x, 2), "y": round(y, 2), "k": kind, "l": layer})

    for e in doc.modelspace():
        dt = e.dxftype()
        if dt == "TEXT":
            p = e.dxf.insert
            add(e.dxf.text, p.x, p.y, "T", e.dxf.layer)
        elif dt == "MTEXT":
            p = e.dxf.insert
            add(e.plain_text(), p.x, p.y, "M", e.dxf.layer)
        elif dt == "INSERT":
            p = e.dxf.insert
            for a in e.attribs:
                add(a.dxf.text, p.x, p.y, f"A:{e.dxf.name}", e.dxf.layer)
    return items


def extract_all(dxf_dir: Path, out_json: Path) -> dict[str, list[dict]]:
    """dxf_dir 전체를 훑어 {파일명: [텍스트...]}를 JSON으로 저장한다."""
    out: dict[str, list[dict]] = {}
    for path in sorted(dxf_dir.glob("*.dxf")):
        out[path.name] = extract_text(path)
        logger.info("%s: 텍스트 %d건", path.name, len(out[path.name]))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def render(dxf_path: Path, out_png: Path, box: tuple[float, float, float, float] | None = None) -> Path:
    """DXF를 PNG로 렌더한다. box=(x0,y0,x1,y1)을 주면 그 영역만 확대한다.

    도면 원본 색은 검은 배경 기준(흰 선)이라 그대로 그리면 흰 배경에 안 보인다 →
    BackgroundPolicy.WHITE + ColorPolicy.BLACK으로 강제 반전한다.
    """
    doc = ezdxf.readfile(dxf_path)
    cfg = draw_config.Configuration(
        background_policy=draw_config.BackgroundPolicy.WHITE,
        color_policy=draw_config.ColorPolicy.BLACK,
        lineweight_policy=draw_config.LineweightPolicy.ABSOLUTE,
        lineweight_scaling=0.5,
    )
    fig = plt.figure(figsize=config.RENDER_FIGSIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=cfg).draw_layout(
        doc.modelspace(), finalize=False
    )
    ax.set_aspect("equal")
    if box:
        ax.set_xlim(box[0], box[2])
        ax.set_ylim(box[1], box[3])
        ax.set_aspect("equal", adjustable="datalim")
    else:
        ax.autoscale()
        ax.autoscale_view()
    ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=config.RENDER_DPI, facecolor="white")
    plt.close(fig)
    return out_png
