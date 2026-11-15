#!/usr/bin/env python3
"""Дочерний процесс озвучки для VoiceTyper: синтез Silero + воспроизведение.

Вынесен в отдельный процесс по той же причине, что и audio_helper.py:
PortAudio/CoreAudio после смены аудиоустройства или выхода из сна может
навсегда завесить любой вызов, и внутри процесса это не лечится. Отдельный
процесс родитель убивает по таймауту и поднимает заново. Заодно тяжёлый
torch с моделью не живёт в одном процессе с Whisper.

Запуск:  tts_helper.py <model.pt> <model-url> <speaker> <sample-rate>

Модель - пакет torch.package от Silero. Если файла нет, он один раз
скачивается по <model-url>; дальше сеть не нужна вовсе.

Протокол (строки через stdin/stdout):

  родитель -> хелпер:  SPEAK <id> <json>          прочитать текст вслух; json - строка
                                                  или {"text": ..., "rate": ...}, где rate -
                                                  темп SSML: x-slow/slow/medium/fast/x-fast
                       STOP                       прервать чтение
  хелпер -> родитель:  READY          модель загружена, можно командовать
                       OK             команда принята
                       DONE <id>      чтение <id> закончилось (дочитал или прервали)
                       LOG <текст>    сообщение для лога родителя
                       ERR <текст>    команда не удалась

SPEAK отвечает OK сразу: синтез и воспроизведение идут в отдельном потоке,
чтобы STOP мог прервать их в любой момент. Новый SPEAK во время чтения
сначала прерывает текущее.

Жизнь без родителя не имеет смысла: EOF на stdin - выходим; смена ppid
ловится сторожевым потоком.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.request
import warnings
from pathlib import Path

# Пакет Silero при импорте сыплет SyntaxWarning про escape-последовательности -
# к делу не относится, а stderr всё равно никто не читает.
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402
import torch  # noqa: E402

# Модель падает на куске длиннее ~1000 символов ("Model couldn't generate
# your text, probably it's too long"; проверено: 900 - ок, 1000 - уже нет).
# Режем по предложениям с запасом: заодно первый звук появляется быстрее.
MAX_CHUNK_CHARS = 500
BLOCK_SECONDS = 0.1  # гранулярность записи в поток: так STOP срабатывает быстро
DOWNLOAD_TIMEOUT = 60.0

_stdout_lock = threading.Lock()


def _send(line: str) -> None:
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _parent_watchdog() -> None:
    parent = os.getppid()
    while True:
        time.sleep(2.0)
        if os.getppid() != parent:
            os._exit(1)


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------


def _ensure_model(path: Path, url: str) -> None:
    """Скачивает модель один раз. Атомарно: недокачанный файл не останется
    лежать под рабочим именем и не сломает следующий запуск."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(path)


def _load_model(path: Path):
    torch.set_num_threads(4)
    importer = torch.package.PackageImporter(str(path))
    model = importer.load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    return model


# ---------------------------------------------------------------------------
# Нарезка текста
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?…;:])\s+|\n+")
# Silero отказывается (ValueError) от куска, в котором нет ни одной русской
# буквы - латиница, цифры и знаки им выбрасываются. Такие куски пропускаем.
_HAS_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def split_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Режет текст на куски не длиннее limit, по возможности по предложениям."""
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        sentence = " ".join(sentence.split())
        if not sentence:
            continue
        # Предложение длиннее лимита режем по словам.
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            piece, sentence = sentence[:cut], sentence[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece)
        if current and len(current) + 1 + len(sentence) > limit:
            chunks.append(current)
            current = ""
        current = f"{current} {sentence}" if current else sentence
    if current:
        chunks.append(current)
    return [c for c in chunks if _HAS_CYRILLIC.search(c)]


# ---------------------------------------------------------------------------
# Чтение
# ---------------------------------------------------------------------------


class Reader:
    """Одно чтение за раз: поток синтеза+воспроизведения и флаг остановки."""

    def __init__(self, model, speaker: str, sample_rate: int) -> None:
        self._model = model
        self._speaker = speaker
        self._rate = sample_rate
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def speak(self, speak_id: str, text: str, rate: str = "medium") -> None:
        self.stop()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(speak_id, text, rate, self._stop), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join()
        self._thread = None

    def _synth(self, chunk: str, rate: str):
        if rate == "medium":
            return self._model.apply_tts(text=chunk, speaker=self._speaker, sample_rate=self._rate)
        escaped = chunk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return self._model.apply_tts(
            ssml_text=f'<speak><prosody rate="{rate}">{escaped}</prosody></speak>',
            speaker=self._speaker,
            sample_rate=self._rate,
        )

    def _run(self, speak_id: str, text: str, rate: str, stop: threading.Event) -> None:
        stream = None
        try:
            block = int(self._rate * BLOCK_SECONDS)
            for chunk in split_text(text):
                if stop.is_set():
                    break
                try:
                    audio = self._synth(chunk, rate)
                except Exception as exc:  # noqa: BLE001 - один кусок не валит чтение
                    _send(f"LOG кусок не синтезирован ({type(exc).__name__}): {chunk[:60]!r}")
                    continue
                samples = audio.numpy().astype(np.float32).reshape(-1, 1)
                if stream is None:
                    stream = sd.OutputStream(samplerate=self._rate, channels=1, dtype="float32")
                    stream.start()
                for start in range(0, samples.shape[0], block):
                    if stop.is_set():
                        break
                    stream.write(samples[start : start + block])
            if stream is not None and not stop.is_set():
                # write() возвращает управление, когда кадры ушли в буфер, а не
                # когда отзвучали: без паузы хвост фразы обрезался бы.
                time.sleep(stream.latency + 0.05)
        except Exception as exc:  # noqa: BLE001
            _send(f"LOG чтение прервано ошибкой: {exc}")
        finally:
            if stream is not None:
                try:
                    stream.abort()
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
            _send(f"DONE {speak_id}")


def main() -> None:
    model_path, url, speaker, rate = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
    threading.Thread(target=_parent_watchdog, daemon=True).start()

    try:
        _ensure_model(model_path, url)
        model = _load_model(model_path)
    except Exception as exc:  # noqa: BLE001 - решает родитель
        _send(f"ERR модель озвучки не загрузилась: {exc}")
        return

    reader = Reader(model, speaker, rate)
    _send("READY")

    for line in sys.stdin:  # EOF = родитель умер - выходим
        cmd, _, arg = line.rstrip("\n").partition(" ")
        try:
            if cmd == "SPEAK":
                speak_id, _, payload = arg.partition(" ")
                data = json.loads(payload)
                if isinstance(data, str):
                    data = {"text": data}
                reader.speak(speak_id, data["text"], data.get("rate") or "medium")
                _send("OK")
            elif cmd == "STOP":
                reader.stop()
                _send("OK")
            else:
                raise RuntimeError(f"неизвестная команда {cmd!r}")
        except Exception as exc:  # noqa: BLE001
            _send(f"ERR {exc}")


if __name__ == "__main__":
    main()
