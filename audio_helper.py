#!/usr/bin/env python3
"""Дочерний процесс записи звука для VoiceTyper.

Захват микрофона вынесен в отдельный процесс, потому что PortAudio/CoreAudio
после смены аудиоустройства или выхода из сна может завесить любой вызов
(Pa_StopStream, Pa_OpenStream, Pa_Terminate) навсегда - причём клин
процесс-глобальный: после первого зависшего вызова виснут и все последующие,
включая переинициализацию. Внутри одного процесса это не лечится ни таймаутами,
ни бросанием зависших потоков (проверено вживую). Отдельный процесс родитель
просто убивает по таймауту и поднимает новый - протухшее CoreAudio-состояние
умирает вместе с ним.

Протокол (строки через stdin/stdout):

  родитель -> хелпер:  START <path>   начать запись сырых int16-кадров в файл
                       STOP           остановить запись
  хелпер -> родитель:  READY          процесс поднялся, можно командовать
                       STARTED        поток открыт, запись пошла
                       LEVEL <0..1>   пиковый уровень с прошлого LEVEL (~30/с)
                       STOPPED <n>    запись остановлена, в файле n кадров
                       ERR <текст>    команда не удалась

Кадры пишутся в файл через os.write сразу из аудио-колбэка, без буферизации:
если хелпер завис и родитель убил его посреди записи, всё записанное до клина
остаётся в файле и годится для распознавания.

Жизнь без родителя не имеет смысла: EOF на stdin - выходим; смена ppid
(родителя убили, а мы висим в C-коде и stdin не читаем) ловится сторожевым
потоком.
"""

import os
import sys
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
LEVEL_INTERVAL = 1.0 / 30

_stdout_lock = threading.Lock()


def _send(line: str) -> None:
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _parent_watchdog() -> None:
    """Если родитель умер, а мы этого не заметили (главный поток завис в
    CoreAudio и stdin не читает) - выходим сами, не оставаясь сиротой."""
    parent = os.getppid()
    while True:
        time.sleep(2.0)
        if os.getppid() != parent:
            os._exit(1)


class _Recording:
    """Одна запись: открытый поток PortAudio + файл с кадрами."""

    def __init__(self, path: str):
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        self.frames = 0
        self._peak = 0.0
        self._done = threading.Event()

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=self._callback
        )
        self.stream.start()
        threading.Thread(target=self._level_sender, daemon=True).start()

    def _callback(self, indata, frame_count, time_info, status):
        os.write(self._fd, indata.tobytes())
        self.frames += frame_count
        peak = float(np.abs(indata).max()) / 32768.0
        if peak > self._peak:
            self._peak = peak

    def _level_sender(self) -> None:
        while not self._done.wait(LEVEL_INTERVAL):
            _send(f"LEVEL {self._peak:.3f}")
            self._peak = 0.0

    def stop(self) -> int:
        self._done.set()
        self.stream.stop()
        self.stream.close()
        os.close(self._fd)
        return self.frames


def main() -> None:
    threading.Thread(target=_parent_watchdog, daemon=True).start()
    _send("READY")
    rec: _Recording | None = None

    for line in sys.stdin:  # EOF = родитель умер - выходим
        cmd, _, arg = line.rstrip("\n").partition(" ")
        try:
            if cmd == "START":
                if rec is not None:
                    raise RuntimeError("запись уже идёт")
                rec = _Recording(arg)
                _send("STARTED")
            elif cmd == "STOP":
                if rec is None:
                    _send("STOPPED 0")
                    continue
                frames, rec = rec.stop(), None
                _send(f"STOPPED {frames}")
            else:
                raise RuntimeError(f"неизвестная команда {cmd!r}")
        except Exception as exc:  # noqa: BLE001 - решает родитель
            rec = None
            _send(f"ERR {exc}")


if __name__ == "__main__":
    main()
