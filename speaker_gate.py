"""Проверка голоса для режима Джарвис: слушаться только владельца.

Каждая фраза, услышанная без кнопки, превращается в «отпечаток голоса»
(speaker embedding, модель CAM++ через sherpa-onnx, офлайн, на CPU) и
сравнивается по косинусу с отпечатком владельца, записанным заранее
(enroll_voice.py). Ниже порога - фраза чужая, Джарвис её не слышит.

Это фильтр, а не защита: запись голоса владельца или очень похожий голос
пройдут. Он убирает случайные срабатывания на телевизор и чужую речь.

Модель (~28 МБ) скачивается один раз в DATA_DIR, туда же ложится отпечаток.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
DATA_DIR = Path.home() / "Library" / "Application Support" / "voice-typer"
MODEL_NAME = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/" + MODEL_NAME
)
VOICEPRINT_PATH = DATA_DIR / "voiceprint.npy"
DOWNLOAD_TIMEOUT = 60.0


def _ensure_model() -> Path:
    path = DATA_DIR / MODEL_NAME
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URL, timeout=DOWNLOAD_TIMEOUT) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(path)
    return path


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


class SpeakerGate:
    """Потокобезопасно: модель грузится лениво, отпечаток перечитывается,
    если файл поменялся (перезаписали голос - приложение перезапускать не надо)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._extractor = None
        self._ref: np.ndarray | None = None
        self._ref_mtime: float | None = None

    def load(self) -> None:
        """Загрузить модель заранее, чтобы первая фраза не ждала."""
        with self._lock:
            self._get_extractor()

    def _get_extractor(self):
        if self._extractor is None:
            import sherpa_onnx

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(_ensure_model()), num_threads=1
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        return self._extractor

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """audio - int16 или float32, 16 кГц, моно."""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        with self._lock:
            ex = self._get_extractor()
            stream = ex.create_stream()
            stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
            stream.input_finished()
            return _normalize(np.asarray(ex.compute(stream), dtype=np.float32))

    def has_voiceprint(self) -> bool:
        return self._voiceprint() is not None

    def _voiceprint(self) -> np.ndarray | None:
        try:
            mtime = VOICEPRINT_PATH.stat().st_mtime
        except FileNotFoundError:
            self._ref = self._ref_mtime = None
            return None
        if mtime != self._ref_mtime:
            self._ref = _normalize(np.load(VOICEPRINT_PATH).astype(np.float32))
            self._ref_mtime = mtime
        return self._ref

    def score(self, audio: np.ndarray) -> float | None:
        """Косинус с отпечатком владельца (1.0 - тот же голос) или None,
        если отпечатка ещё нет."""
        ref = self._voiceprint()
        if ref is None:
            return None
        return float(self.embed(audio) @ ref)


def save_voiceprint(embeddings: list[np.ndarray]) -> np.ndarray:
    ref = _normalize(np.mean(np.stack(embeddings), axis=0))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = VOICEPRINT_PATH.with_name("voiceprint.tmp.npy")
    np.save(tmp, ref)
    os.replace(tmp, VOICEPRINT_PATH)
    return ref
