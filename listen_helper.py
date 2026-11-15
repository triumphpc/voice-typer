#!/usr/bin/env python3
"""Дочерний процесс «слушаю без кнопки» для режима Джарвис.

Держит микрофон открытым, сам находит начало и конец фразы и отдаёт
родителю готовую фразу файлом. Отдельный процесс - по той же причине, что и
audio_helper.py: зависший CoreAudio лечится только убийством процесса. Заодно
микрофон освобождается (гаснет оранжевая точка), как только режим выключили
и родитель убил хелпер.

Детектор речи - по громкости, без модели: порог плавает вслед за уровнем
шума в комнате. Ложные срабатывания на стук или кашель отсекает уже
родитель: Whisper вернёт пустоту или фразу из списка галлюцинаций.

Запуск:  listen_helper.py <каталог для фраз>

Протокол (строки через stdin/stdout):

  родитель -> хелпер:  MUTE           не слушать (Джарвис думает или говорит)
                       UNMUTE         слушать снова
  хелпер -> родитель:  READY          микрофон открыт
                       SPEECH         началась фраза
                       LEVEL <0..1>   уровень во время фразы (~30/с), для панели
                       UTT <path> <s> фраза закончилась: сырые int16-кадры, 16 кГц
                       DROP           фраза оказалась шумом - забыли
                       LOG <текст>    сообщение для лога родителя
                       ERR <текст>    не удалось открыть микрофон

После UTT хелпер сам перестаёт слушать: собственный ответ Джарвиса из колонок
не должен стать следующим вопросом. Слушать снова - только после UNMUTE.
"""

import collections
import os
import sys
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
FRAME = 480  # 30 мс
FRAME_SEC = FRAME / SAMPLE_RATE

START_FRAMES = 4  # из последних START_WINDOW кадров громких должно быть столько
START_WINDOW = 6
END_SILENCE_SEC = 0.7  # тишина после речи, после которой фраза считается законченной
PREROLL_SEC = 0.3  # сколько звука до срабатывания приложить к фразе (начало слова)
MIN_VOICED_SEC = 0.35  # меньше громкого звука - это щелчок, а не фраза
MAX_UTTERANCE_SEC = 30.0
UNMUTE_GUARD_SEC = 0.35  # после UNMUTE не слушаем: хвост собственного голоса в колонках

NOISE_ALPHA = 0.03  # скорость, с которой порог шума следует за тишиной
NOISE_MIN = 0.0015
SPEECH_RATIO = 3.0  # громче шума во столько раз - речь
SPEECH_MIN_RMS = 0.006

DEBUG = os.environ.get("VT_LISTEN_DEBUG") == "1"  # раз в 3 с слать STAT с уровнями

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


class Endpointer:
    """Режет поток 30-мс кадров на фразы. Без ввода-вывода - чтобы проверять
    на записанном звуке. feed() возвращает событие: "speech", готовую фразу
    (np.ndarray int16), "drop" или None."""

    def __init__(self) -> None:
        self.noise = 0.004
        self.max_rms = 0.0  # для диагностики (STAT)
        self.reset()

    def reset(self) -> None:
        self._recent = collections.deque(maxlen=START_WINDOW)
        self._preroll = collections.deque(maxlen=int(PREROLL_SEC / FRAME_SEC) + START_WINDOW)
        self._utt: list[np.ndarray] | None = None
        self._voiced = 0
        self._silent = 0

    @property
    def in_speech(self) -> bool:
        return self._utt is not None

    def _is_loud(self, rms: float) -> bool:
        return rms > max(self.noise * SPEECH_RATIO, SPEECH_MIN_RMS)

    def feed(self, frame: np.ndarray):
        rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
        self.max_rms = max(self.max_rms, rms)
        loud = self._is_loud(rms)

        if self._utt is None:
            if not loud:
                # Порог следует за тишиной вниз и вверх (включили кондиционер).
                self.noise += NOISE_ALPHA * (rms - self.noise)
                self.noise = max(self.noise, NOISE_MIN)
            self._recent.append(loud)
            self._preroll.append(frame)
            if sum(self._recent) >= START_FRAMES:
                self._utt = list(self._preroll)
                self._voiced = sum(self._recent)
                self._silent = 0
                return "speech"
            return None

        self._utt.append(frame)
        if loud:
            self._voiced += 1
            self._silent = 0
        else:
            self._silent += 1

        too_long = len(self._utt) * FRAME_SEC >= MAX_UTTERANCE_SEC
        if self._silent * FRAME_SEC < END_SILENCE_SEC and not too_long:
            return None

        frames, voiced = self._utt, self._voiced
        self.reset()
        if voiced * FRAME_SEC < MIN_VOICED_SEC:
            return "drop"
        # Хвост тишины Whisper'у не нужен - оставляем чуть-чуть.
        keep = len(frames) - max(0, self._tail_to_cut(frames))
        return np.concatenate(frames[:keep])

    @staticmethod
    def _tail_to_cut(frames) -> int:
        return int(END_SILENCE_SEC / FRAME_SEC) - int(0.2 / FRAME_SEC)


class Listener:
    def __init__(self, out_dir: str) -> None:
        self._out_dir = out_dir
        self._ep = Endpointer()
        self._lock = threading.Lock()
        self._muted = True
        self._listen_after = 0.0
        self._peak = 0.0
        self._n = 0
        self._buf = np.zeros(0, dtype=np.int16)
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
            callback=self._callback,
        )
        self.stream.start()
        threading.Thread(target=self._level_sender, daemon=True).start()

    def mute(self) -> None:
        with self._lock:
            self._muted = True
            self._ep.reset()

    def unmute(self) -> None:
        with self._lock:
            self._muted = False
            self._listen_after = time.time() + UNMUTE_GUARD_SEC
            self._ep.reset()

    def _callback(self, indata, frame_count, time_info, status):
        # Исключение из аудио-колбэка молча останавливает поток PortAudio:
        # микрофон «оглох бы» без единого сообщения. Поэтому - ловим и пишем.
        try:
            self._process(indata[:, 0])
        except Exception as exc:  # noqa: BLE001
            _send(f"LOG ошибка в обработке звука: {type(exc).__name__}: {exc}")
            self._ep.reset()

    def _process(self, samples: np.ndarray) -> None:
        with self._lock:
            if self._muted or time.time() < self._listen_after:
                self._buf = np.zeros(0, dtype=np.int16)
                return
            self._buf = np.concatenate([self._buf, samples])
            while self._buf.size >= FRAME:
                frame, self._buf = self._buf[:FRAME], self._buf[FRAME:]
                event = self._ep.feed(frame)
                if self._ep.in_speech:
                    self._peak = max(self._peak, float(np.abs(frame).max()) / 32768.0)
                if event is None:
                    continue
                if isinstance(event, np.ndarray):
                    self._emit(event)
                    self._muted = True  # ждём UNMUTE от родителя
                    self._buf = np.zeros(0, dtype=np.int16)
                    return
                _send("SPEECH" if event == "speech" else "DROP")

    def _emit(self, audio: np.ndarray) -> None:
        self._n += 1
        path = os.path.join(self._out_dir, f"utt-{os.getpid()}-{self._n}.raw")
        with open(path, "wb") as f:
            f.write(audio.tobytes())
        _send(f"UTT {path} {audio.size / SAMPLE_RATE:.2f}")

    def _level_sender(self) -> None:
        last_stat = time.time()
        while True:
            time.sleep(1.0 / 30)
            if DEBUG and time.time() - last_stat > 3.0:
                last_stat = time.time()
                _send(f"STAT muted={self._muted} noise={self._ep.noise:.4f} max_rms={self._ep.max_rms:.4f}")
                self._ep.max_rms = 0.0
            if self._ep.in_speech:
                _send(f"LEVEL {self._peak:.3f}")
                self._peak = 0.0


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.gettempdir()
    threading.Thread(target=_parent_watchdog, daemon=True).start()
    try:
        listener = Listener(out_dir)
    except Exception as exc:  # noqa: BLE001 - решает родитель
        _send(f"ERR микрофон не открылся: {exc}")
        return
    _send("READY")

    for line in sys.stdin:  # EOF = родитель умер - выходим
        cmd = line.strip()
        if cmd == "MUTE":
            listener.mute()
        elif cmd == "UNMUTE":
            listener.unmute()


if __name__ == "__main__":
    main()
