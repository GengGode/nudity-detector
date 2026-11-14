"""
HTTP API for the moderation pipeline. Your app posts an image, gets back JSON.

Endpoints:
  GET  /health         -> {"status":"ok","ready":true}
  POST /check          -> moderation verdict for one image (multipart upload)
  GET  /policy         -> the active policy thresholds

Run:
  Windows:  run-server.bat
  Manual:   .venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000

The first request triggers model loading (~10s). Set up a process manager
(systemd, pm2, supervisor, Windows Service) to keep this running.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from analyzer import CombinedAnalyzer, VerdictPolicy
from detector import EXPLICIT_CLASSES, Detector
from nsfw_classifier import NSFWClassifier


# ---------------------------------------------------------------------------
# Production policy — these are the values you "locked in" via the UI.
# Tune via env vars without code changes.
# ---------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


PRODUCTION_RESOLUTION = _env_int("MOD_INFERENCE_RESOLUTION", 1024)
PRODUCTION_DISPLAY_THRESHOLD = _env_float("MOD_DISPLAY_THRESHOLD", 0.15)
# Resolutions the detector can be instantiated at. 1280 is the useful
# "hard cases" setting; 320/640 are cheaper debug/preview modes.
ALLOWED_RESOLUTIONS = {320, 640, 1024, 1280}
PRODUCTION_POLICY = VerdictPolicy(
    nsfw_block_threshold=_env_float("MOD_NSFW_BLOCK", 0.70),
    nsfw_review_threshold=_env_float("MOD_NSFW_REVIEW", 0.30),
    explicit_block_threshold=_env_float("MOD_EXPLICIT_BLOCK", 0.50),
    explicit_review_threshold=_env_float("MOD_EXPLICIT_REVIEW", 0.20),
    male_score_boost=_env_float("MOD_MALE_BOOST", 2.5),
    # A face of one gender used to delete the other gender's explicit detections, so a
    # woman's face next to exposed male genitals could pass. Off: an extra REVIEW is cheaper.
    drop_male_when_female_face=_env_bool("MOD_DROP_BY_FACE", False),
    drop_female_when_male_face=_env_bool("MOD_DROP_BY_FACE", False),
    # Above 1.0 no two boxes ever "conflict", so overlapping detections are never both dropped.
    conflict_iou_threshold=_env_float("MOD_CONFLICT_IOU", 1.01),
)
MAX_IMAGE_BYTES = _env_int("MOD_MAX_IMAGE_BYTES", 15 * 1024 * 1024)  # 15 MB
# The checked pixels must be the shown pixels: only formats without transparency or extra
# frames (JPEG by default), and no side longer than this. A tiny file can declare a huge
# canvas (30000x1), and padding it to a square used to take gigabytes of memory.
ALLOWED_FORMATS = {f.strip().upper() for f in os.environ.get("MOD_FORMATS", "JPEG").split(",") if f.strip()}
MAX_SIDE = _env_int("MOD_MAX_SIDE", 2048)
Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("moderation")

if PRODUCTION_RESOLUTION not in ALLOWED_RESOLUTIONS:
    log.warning(
        "MOD_INFERENCE_RESOLUTION=%s is not one of %s; falling back to 1024",
        PRODUCTION_RESOLUTION,
        sorted(ALLOWED_RESOLUTIONS),
    )
    PRODUCTION_RESOLUTION = 1024


# ---------------------------------------------------------------------------
# Model cache: one NudeNet session per inference resolution, one shared
# Falconsai classifier. The 1024 production analyzer is loaded at startup;
# other resolutions are loaded lazily on first use and stay cached.
# ---------------------------------------------------------------------------
_analyzers: dict[int, CombinedAnalyzer] = {}
_classifier: NSFWClassifier | None = None
_analyzer_lock = threading.RLock()


def _get_classifier() -> NSFWClassifier:
    global _classifier
    with _analyzer_lock:
        if _classifier is None:
            log.info("Loading NSFW classifier (Falconsai)")
            _classifier = NSFWClassifier()
        return _classifier


def _get_analyzer(resolution: int) -> CombinedAnalyzer:
    resolution = int(resolution)
    if resolution not in ALLOWED_RESOLUTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}",
        )
    with _analyzer_lock:
        analyzer = _analyzers.get(resolution)
        if analyzer is None:
            log.info("Loading detector (NudeNet 640m) at resolution=%d", resolution)
            detector = Detector(
                inference_resolution=resolution,
                score_threshold=0.05,
                nms_threshold=0.05,
            )
            analyzer = CombinedAnalyzer(
                detector=detector,
                classifier=_get_classifier(),
                policy=PRODUCTION_POLICY,
            )
            _analyzers[resolution] = analyzer
        return analyzer


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_analyzer(PRODUCTION_RESOLUTION)
    log.info("Ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="ModerationKit API",
    version="1.0",
    lifespan=lifespan,
)

# Adjust origins to your app's domain in production for browser-based callers.
# Default is permissive because most callers are server-side.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MOD_ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DetectionDTO(BaseModel):
    cls: str = Field(..., alias="class")
    category: str
    score: float
    box: list[int]

    model_config = {"populate_by_name": True}


class CheckResponse(BaseModel):
    verdict: str  # ALLOW | REVIEW | BLOCK
    reason: str
    nsfw_score: float
    safe_score: float
    explicit_detections: list[DetectionDTO]
    all_detections: list[DetectionDTO]
    conflicts: list[str]
    elapsed_ms: int
    model_versions: dict[str, str]
    # Echo the request-scoped settings so callers can log exactly what ran.
    inference_resolution: int | None = None
    display_threshold: float | None = None
    policy: dict[str, float] | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    inference_resolution: int


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _ensure_ready(resolution: int) -> CombinedAnalyzer:
    if not _analyzers:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return _get_analyzer(resolution)


def _validate_image_bytes(data: bytes) -> Image.Image:
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty image body")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit",
        )
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        # verify() exhausts the stream — re-open for actual use
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image") from None
    if img.format not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail=f"Only {', '.join(sorted(ALLOWED_FORMATS))} images are accepted")
    if max(img.size) > MAX_SIDE or getattr(img, "n_frames", 1) != 1:
        raise HTTPException(status_code=413, detail=f"Images must be one frame, at most {MAX_SIDE} pixels a side")
    return img.convert("RGB")


def _unit_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"{name} must be a number") from None
    if not 0.0 <= number <= 1.0:
        raise HTTPException(status_code=422, detail=f"{name} must be between 0 and 1")
    return number


def _resolve_request_settings(
    *,
    resolution: int | None,
    display_threshold: float | None,
    nsfw_block_threshold: float | None,
    nsfw_review_threshold: float | None,
    explicit_block_threshold: float | None,
    explicit_review_threshold: float | None,
    male_score_boost: float | None,
) -> tuple[int, VerdictPolicy, float]:
    """Apply optional per-request overrides on top of the production policy."""
    if resolution is None:
        resolution = PRODUCTION_RESOLUTION
    try:
        resolution = int(resolution)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="resolution must be an integer") from None
    if resolution not in ALLOWED_RESOLUTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}",
        )

    changes: dict[str, float] = {}
    for name, value in (
        ("nsfw_block_threshold", nsfw_block_threshold),
        ("nsfw_review_threshold", nsfw_review_threshold),
        ("explicit_block_threshold", explicit_block_threshold),
        ("explicit_review_threshold", explicit_review_threshold),
    ):
        parsed = _unit_float(name, value)
        if parsed is not None:
            changes[name] = parsed

    if male_score_boost is not None:
        try:
            boost = float(male_score_boost)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="male_score_boost must be a number") from None
        if not 1.0 <= boost <= 4.0:
            raise HTTPException(status_code=422, detail="male_score_boost must be between 1 and 4")
        changes["male_score_boost"] = boost

    policy = replace(PRODUCTION_POLICY, **changes) if changes else PRODUCTION_POLICY
    display = _unit_float("display_threshold", display_threshold)
    if display is None:
        display = PRODUCTION_DISPLAY_THRESHOLD
    return resolution, policy, display


def _analyze_pil(
    img: Image.Image,
    *,
    resolution: int,
    policy: VerdictPolicy,
    display_threshold: float,
) -> CheckResponse:
    a = _ensure_ready(resolution)
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        img.save(tmp_path, format="JPEG", quality=92)
        result = a.analyze(tmp_path, policy=policy)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    elapsed_ms = int((time.time() - t0) * 1000)

    explicit = [
        DetectionDTO(
            **{"class": d.class_name},
            category=d.category,
            score=round(d.score, 4),
            box=list(d.box),
        )
        for d in result.kept_detections
        if d.class_name in EXPLICIT_CLASSES and d.score >= display_threshold
    ]
    all_dets = [
        DetectionDTO(
            **{"class": d.class_name},
            category=d.category,
            score=round(d.score, 4),
            box=list(d.box),
        )
        for d in result.kept_detections
        if d.score >= display_threshold
    ]

    return CheckResponse(
        verdict=result.verdict,
        reason=result.reason,
        nsfw_score=round(result.nsfw_score.nsfw, 4),
        safe_score=round(result.nsfw_score.normal, 4),
        explicit_detections=explicit,
        all_detections=all_dets,
        conflicts=result.conflicts,
        elapsed_ms=elapsed_ms,
        model_versions={
            "detector": "nudenet-640m",
            "classifier": "Falconsai/nsfw_image_detection",
        },
        inference_resolution=resolution,
        display_threshold=display_threshold,
        policy={
            "nsfw_block_threshold": policy.nsfw_block_threshold,
            "nsfw_review_threshold": policy.nsfw_review_threshold,
            "explicit_block_threshold": policy.explicit_block_threshold,
            "explicit_review_threshold": policy.explicit_review_threshold,
            "male_score_boost": policy.male_score_boost,
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ready=bool(_analyzers),
        inference_resolution=PRODUCTION_RESOLUTION,
    )


@app.get("/policy")
def policy() -> dict:
    return {
        "inference_resolution": PRODUCTION_RESOLUTION,
        "display_threshold": PRODUCTION_DISPLAY_THRESHOLD,
        "max_image_bytes": MAX_IMAGE_BYTES,
        "verdict_policy": {
            "nsfw_block_threshold": PRODUCTION_POLICY.nsfw_block_threshold,
            "nsfw_review_threshold": PRODUCTION_POLICY.nsfw_review_threshold,
            "explicit_block_threshold": PRODUCTION_POLICY.explicit_block_threshold,
            "explicit_review_threshold": PRODUCTION_POLICY.explicit_review_threshold,
            "male_score_boost": PRODUCTION_POLICY.male_score_boost,
        },
    }


@app.post("/check", response_model=CheckResponse)
async def check(
    file: UploadFile = File(...),
    resolution: int | None = Form(default=None),
    display_threshold: float | None = Form(default=None),
    nsfw_block_threshold: float | None = Form(default=None),
    nsfw_review_threshold: float | None = Form(default=None),
    explicit_block_threshold: float | None = Form(default=None),
    explicit_review_threshold: float | None = Form(default=None),
    male_score_boost: float | None = Form(default=None),
) -> CheckResponse:
    """
    Multipart upload:  field name = `file`.

    Optional request-scoped overrides (kept for backward compatibility):
      resolution               320 | 640 | 1024 | 1280
      display_threshold        0..1 (filters returned detections only)
      nsfw_block_threshold     0..1
      nsfw_review_threshold    0..1
      explicit_block_threshold 0..1
      explicit_review_threshold 0..1
      male_score_boost         1..4

    Omitting any field keeps the production default from env/policy.
    """
    resolution, policy, display_threshold = _resolve_request_settings(
        resolution=resolution,
        display_threshold=display_threshold,
        nsfw_block_threshold=nsfw_block_threshold,
        nsfw_review_threshold=nsfw_review_threshold,
        explicit_block_threshold=explicit_block_threshold,
        explicit_review_threshold=explicit_review_threshold,
        male_score_boost=male_score_boost,
    )
    data = await file.read()
    img = _validate_image_bytes(data)
    return _analyze_pil(
        img,
        resolution=resolution,
        policy=policy,
        display_threshold=display_threshold,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
