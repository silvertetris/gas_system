"""조합별 비교 실행기 — 구조(hard·soft·nn) × 타깃(min_raw·min·mean) × 정제 조합 × 계열 M·Z.

정제 조합: 넣기 전(없음) / KF만 / AE만 / EKF만 / 셋 다. 조합마다 **따로** Optuna(같은 개발구간 5-fold CV,
같은 탐색공간·trial 수·표본기 시드)로 하이퍼파라미터를 찾고, 최종 학습을 시드 5개로 반복해 시험구간 20% 에서
짝비교한다. 작업마다 XAI·진단 그림도 만든다.

    PYTHONPATH=. python -m pinn.restricted.variants_run --workers 3                        # 기존 hard·min_raw
    PYTHONPATH=. python -m pinn.restricted.variants_run --archs hard soft nn --targets min mean --done-name matrix_DONE

작업은 별도 프로세스로 병렬 실행한다. 이미 끝난 작업(산출 파일 존재)은 건너뛰므로 중단 후 다시 실행하면
이어서 돈다. 순서는 정제 조합 우선순위(없음 → 셋 다 → KF → AE → EKF) 단위로, 구조·타깃 비교가 조합별로
먼저 완성되게 한다.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time

from . import config, tune

log = logging.getLogger("pinn.restricted.variants")
SEEDS = [42, 7, 123, 2024, 31337]
PRIORITY = ["없음", "KF+AE+EKF", "KF", "AE", "EKF"]


def _job(variant: str, train: str, seeds: list[int], arch: str, target: str) -> str | None:
    d = config.variant_dir(variant, False, arch, target)
    py = sys.executable
    tail = ["--variant", variant, "--trains", train, "--arch", arch, "--target", target]
    steps = []
    if not (d / f"metrics_seeds_{train}.csv").exists():
        steps.append([py, "-m", "pinn.restricted.optuna_pipeline", *tail, "--seeds", *map(str, seeds)])
    if not (d / f"perm_groups_{train}.csv").exists():
        steps.append([py, "-m", "pinn.restricted.xai", *tail])
    if not (d / f"xai_{train}.png").exists():
        steps.append([py, "-m", "pinn.restricted.xai_plots", *tail])
    if not (d / f"diag_{train}.png").exists():
        steps.append([py, "-m", "pinn.restricted.diag_plots", *tail])
    if not (d / f"timeline_{train}.png").exists():
        steps.append([py, "-m", "pinn.restricted.timeline_plots", *tail])
    if not steps:
        return None
    return " && ".join(" ".join(f"'{a}'" for a in s) for s in steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--variants", nargs="+", default=PRIORITY, choices=tune.VARIANTS)
    ap.add_argument("--archs", nargs="+", default=["hard"], choices=config.ARCHS)
    ap.add_argument("--targets", nargs="+", default=["min_raw"], choices=config.TARGETS)
    ap.add_argument("--trains", nargs="+", default=list(config.TRAINS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--done-name", default="variants_DONE")
    args = ap.parse_args()
    config.OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(config.OPTUNA_DIR / "variants_run.log", encoding="utf-8")])
    env = os.environ.copy()
    threads = str(max(1, (os.cpu_count() or 4) // args.workers))
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, PYTHONPATH=str(config.PROJECT_ROOT))

    variants = sorted(args.variants, key=PRIORITY.index)
    queue = []
    for v in variants:
        for arch in args.archs:
            for target in args.targets:
                for t in args.trains:
                    cmd = _job(v, t, args.seeds, arch, target)
                    label = f"{arch}·{target}·{v}·{t}"
                    if cmd is None:
                        log.info("건너뜀(완료): %s", label)
                    else:
                        queue.append((label, config.variant_dir(v, False, arch, target), t, cmd))
    log.info("해상도 %d분 · 작업 %d개 · 동시 %d · 스레드 %s · 시드 %s · 구조 %s · 타깃 %s",
             config.RES_MIN, len(queue), args.workers, threads, args.seeds, args.archs, args.targets)

    running: list = []
    failed = []
    while queue or running:
        while queue and len(running) < args.workers:
            label, d, t, cmd = queue.pop(0)
            d.mkdir(parents=True, exist_ok=True)
            fh = open(d / f"run_{t}.log", "a", encoding="utf-8")
            p = subprocess.Popen(["bash", "-c", cmd], stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 cwd=config.PROJECT_ROOT)
            running.append((label, d, t, p, time.time()))
            log.info("시작: %s (pid %d) · 남은 %d", label, p.pid, len(queue))
        time.sleep(20)
        for item in list(running):
            label, d, t, p, t0 = item
            if p.poll() is None:
                continue
            running.remove(item)
            mins = (time.time() - t0) / 60
            if p.returncode == 0:
                log.info("완료: %s (%.0f분)", label, mins)
            else:
                failed.append(label)
                log.error("실패: %s (코드 %d, %.0f분) — %s", label, p.returncode, mins, d / f"run_{t}.log")

    try:
        from . import variants_compare
        variants_compare.main()
    except Exception as e:                       # 비교 그림 실패가 실행 결과를 가리지 않게
        log.exception("비교 요약 실패: %s", e)
    if failed:
        log.error("실패 작업: %s", failed)
    (config.OPTUNA_DIR / args.done_name).write_text("failed=" + str(failed))


if __name__ == "__main__":
    main()
