#!/usr/bin/env python3
"""
Локальный push-to-talk voice typer для macOS.
Зажал Fn -> говоришь -> отпустил -> текст распознаётся Whisper (offline, MLX)
и вставляется в текущее окно через буфер обмена + Cmd+V.

Никакие данные никуда не отправляются - всё работает локально.

Живёт в статус-баре (иконка микрофона). Логи - ~/Library/Logs/VoiceTyper.log.
"""

import faulthandler
import collections
import logging
import logging.handlers
import fcntl
import json
import math
import os
import signal
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import AppKit
import Foundation
import Quartz
from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementCreateApplication,
    AXUIElementPerformAction,
    AXUIElementSetAttributeValue,
    AXUIElementSetMessagingTimeout,
    kAXErrorSuccess,
    kAXFocusedUIElementAttribute,
    kAXSelectedTextAttribute,
)
from Foundation import NSObject, NSOperationQueue

# ---------------------------------------------------------------------------
# Настройки - меняй под себя
# ---------------------------------------------------------------------------

# Хоткей - клавиша Fn (флаг модификатора, а не обычная клавиша, поэтому
# ловится напрямую через Quartz CGEventTap, а не через pynput). На клавиатуре
# MacBook она физически одна, слева снизу - "левый/правый" тут не различить,
# сработает на единственную Fn.

# Модель распознавания (скачается автоматически при первом запуске).
# Варианты (от быстрой/слабой к медленной/точной):
#   mlx-community/whisper-tiny
#   mlx-community/whisper-base
#   mlx-community/whisper-small
#   mlx-community/whisper-large-v3-turbo   <- по умолчанию, хороший баланс на M-чипах
MODEL = "mlx-community/whisper-large-v3-turbo"

# Язык распознавания. "ru" - только русский (быстрее и точнее).
# None - автоопределение языка.
LANGUAGE = "ru"

SAMPLE_RATE = 16000
MIN_RECORDING_SECONDS = 0.4

# Страховка от "залипшей" Fn: если отпускание клавиши потерялось (система
# отключила перехват, ушла в сон, перехватило другое приложение), запись
# оборвётся сама, а не будет писать в память бесконечно.
MAX_RECORDING_SECONDS = 120.0

# Как часто сторож проверяет, что перехват событий ещё жив.
TAP_CHECK_INTERVAL = 2.0

# Потолок на команду аудио-хелперу (старт/стоп записи, запуск процесса).
# PortAudio в хелпере после смены аудиоустройства или выхода из сна может
# заблокироваться в CoreAudio навсегда - по этому таймауту хелпер убивается
# и поднимается заново.
HELPER_OP_TIMEOUT = 5.0

# Индикация записи. Панель с уровнем звука нагляднее звуков и не мешает,
# когда пишешь в наушниках или на созвоне.
SHOW_HUD = True
HUD_BARS = 27
SOUND_FEEDBACK = False

# Иконка в Dock. По умолчанию выключена: место такой утилиты - строка меню,
# а Dock-иконка ещё и забирает фокус при запуске. Включи, если хочешь закрывать
# приложение правым кликом по Dock и видеть его в ⌘-Tab.
SHOW_IN_DOCK = False

# Физический код клавиши V (kVK_ANSI_V). Не зависит от раскладки клавиатуры.
V_KEYCODE = 9

# --- Озвучка выделенного текста (TTS) ---------------------------------------
# Выделил текст, нажал ⌃S - он читается вслух. Повторное нажатие во время
# чтения останавливает его. Синтез локальный: Silero, на CPU, без сети.
TTS_ENABLED = True
# Голоса: aidar, eugene (мужские), baya, kseniya, xenia (женские).
TTS_SPEAKER = "aidar"
TTS_SAMPLE_RATE = 48000  # 8000 / 24000 / 48000
TTS_MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_6_ru.pt"
# Скачивается один раз (~150 МБ) при первом запуске, дальше сеть не нужна.
TTS_MODEL_PATH = Path.home() / ".cache" / "voice-typer" / "v5_6_ru.pt"
# Хоткей: физический код клавиши (kVK_ANSI_S = 1, от раскладки не зависит)
# и обязательные модификаторы. Лишние модификаторы (⌘, ⇧) - не наш хоткей.
TTS_KEYCODE = 1
TTS_MODIFIERS = Quartz.kCGEventFlagMaskControl
TTS_HOTKEY_LABEL = "⌃S"
# Предохранитель от случайного ⌘A на огромном документе.
TTS_MAX_CHARS = 20000
# Старт хелпера озвучки: импорт torch + загрузка модели, а в первый раз ещё
# и скачивание - поэтому таймаут куда щедрее, чем у аудио-хелпера.
TTS_START_TIMEOUT = 300.0
# Диагностика: писать в лог каждое нажатие клавиши хоткея, которое видит
# перехват (тип события, флаги модификаторов).
TTS_DEBUG_KEYS = False

LOG_PATH = Path.home() / "Library" / "Logs" / "VoiceTyper.log"

# Фразы, которые Whisper выдумывает на тишине (наследие субтитров в обучающих
# данных). Сравнение идёт в нижнем регистре, без хвостовой пунктуации.
HALLUCINATIONS = {
    "продолжение следует",
    "субтитры сделал dimatorzok",
    "субтитры создавал dimatorzok",
    "редактор субтитров а.синецкая корректор а.егорова",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "спасибо",
    "продолжение в следующей серии",
    "thank you",
    "thanks for watching",
    "you",
}

# ---------------------------------------------------------------------------
# Логи
# ---------------------------------------------------------------------------

log = logging.getLogger("voice-typer")
_dump_file = None


def _setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    log.addHandler(stderr_handler)

    log.setLevel(logging.INFO)
    log.propagate = False

    # Иначе падение рабочего потока уходит в stderr, которого у приложения нет,
    # и отказ выглядит как "просто перестало работать".
    def _thread_hook(args):
        log.error(
            "поток %s упал: %s",
            args.thread.name if args.thread else "?",
            args.exc_value,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    def _main_hook(exc_type, exc_value, tb):
        log.error("необработанная ошибка: %s", exc_value, exc_info=(exc_type, exc_value, tb))

    threading.excepthook = _thread_hook
    sys.excepthook = _main_hook

    # `kill -USR1 <pid>` вываливает стеки всех потоков в
    # ~/Library/Logs/VoiceTyper.threads.log - единственный способ понять,
    # где приложение зависло, когда у него нет консоли.
    global _dump_file
    _dump_file = open(LOG_PATH.with_name("VoiceTyper.threads.log"), "a")
    faulthandler.enable(file=_dump_file, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=_dump_file, all_threads=True)


# ---------------------------------------------------------------------------
# Состояния и индикация
# ---------------------------------------------------------------------------

IDLE, RECORDING, WORKING, BROKEN = "idle", "recording", "working", "broken"
SPEAKING = "speaking"

# Иконка в строке меню: SF Symbol, а не эмодзи. Эмодзи в строке меню рисуется
# цветным глифом непредсказуемой ширины и плохо читается на тёмном фоне;
# template-изображение macOS перекрашивает под тему сама.
MENUBAR_SYMBOLS = {
    IDLE: ("mic", "🎙"),
    RECORDING: ("mic.fill", "🔴"),
    WORKING: ("waveform", "…"),
    BROKEN: ("exclamationmark.triangle", "⚠️"),
    SPEAKING: ("speaker.wave.2", "🔊"),
}

_status_item = None
_status_line = None
_menu_target = None
_state = IDLE


def _on_main(fn) -> None:
    """Всё, что трогает UI, обязано выполняться на главном потоке."""
    NSOperationQueue.mainQueue().addOperationWithBlock_(fn)


def _apply_menubar_icon(state: str) -> None:
    if _status_item is None:
        return
    button = _status_item.button()
    symbol, fallback = MENUBAR_SYMBOLS[state]

    image = None
    if hasattr(AppKit.NSImage, "imageWithSystemSymbolName_accessibilityDescription_"):
        image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            symbol, None
        )
    if image is not None:
        image.setTemplate_(True)
        button.setImage_(image)
        button.setTitle_("")
    else:  # SF Symbols нет - откатываемся на эмодзи
        button.setImage_(None)
        button.setTitle_(fallback)


def set_status(state: str, text: str) -> None:
    """Единая точка индикации: строка меню + всплывающая панель."""
    global _state
    _state = state

    def apply():
        _apply_menubar_icon(state)
        if _status_line is not None:
            _status_line.setTitle_(text)
        _hud_apply(state)

    _on_main(apply)


class MenuTarget(NSObject):
    def openLog_(self, sender):
        subprocess.Popen(["open", "-a", "Console", str(LOG_PATH)])

    def restartTap_(self, sender):
        threading.Thread(target=reinstall_tap, name="menu-restart", daemon=True).start()

    def stopSpeaking_(self, sender):
        _tts_cmd.put("stop")

    def quitApp_(self, sender):
        log.info("quit from menu")
        AppKit.NSApp().terminate_(None)


def _build_status_item() -> None:
    global _status_item, _status_line, _menu_target

    _menu_target = MenuTarget.alloc().init()

    _status_item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(
        AppKit.NSVariableStatusItemLength
    )
    # Явно и настойчиво: элемент строки меню может быть скрыт с прошлого запуска,
    # macOS помнит это состояние по autosave-имени.
    _status_item.setVisible_(True)
    _apply_menubar_icon(WORKING)

    menu = AppKit.NSMenu.alloc().init()

    _status_line = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "загрузка модели…", None, ""
    )
    _status_line.setEnabled_(False)
    menu.addItem_(_status_line)

    hint = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Зажми Fn, говори, отпусти", None, ""
    )
    hint.setEnabled_(False)
    menu.addItem_(hint)

    if TTS_ENABLED:
        tts_hint = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"Выдели текст, нажми {TTS_HOTKEY_LABEL} - прочитаю вслух", None, ""
        )
        tts_hint.setEnabled_(False)
        menu.addItem_(tts_hint)

    menu.addItem_(AppKit.NSMenuItem.separatorItem())
    if TTS_ENABLED:
        menu.addItem_(
            AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Остановить чтение", "stopSpeaking:", ""
            )
        )
    menu.addItem_(
        AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Показать лог", "openLog:", ""
        )
    )
    menu.addItem_(
        AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Перезапустить перехват Fn", "restartTap:", ""
        )
    )
    menu.addItem_(AppKit.NSMenuItem.separatorItem())
    menu.addItem_(
        AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Выйти", "quitApp:", "q"
        )
    )

    for i in range(menu.numberOfItems()):
        menu.itemAtIndex_(i).setTarget_(_menu_target)

    _status_item.setMenu_(menu)
    log.info("иконка в строке меню создана (visible=%s)", bool(_status_item.isVisible()))


def _build_main_menu() -> None:
    """Меню приложения - чтобы работал ⌘Q, когда VoiceTyper впереди."""
    main = AppKit.NSMenu.alloc().init()
    app_item = AppKit.NSMenuItem.alloc().init()
    main.addItem_(app_item)

    app_menu = AppKit.NSMenu.alloc().init()
    app_menu.addItem_(
        AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Выйти из VoiceTyper", "terminate:", "q"
        )
    )
    app_item.setSubmenu_(app_menu)
    AppKit.NSApp().setMainMenu_(main)


# ---------------------------------------------------------------------------
# Всплывающая панель с уровнем звука
#
# Показывает, что идёт запись, без звукового "бульканья": бегущая дорожка
# уровня микрофона во время записи и пульсирующие точки во время распознавания.
# Панель не активируется и не перехватывает клики - фокус остаётся в том поле,
# куда пойдёт текст.
# ---------------------------------------------------------------------------

_hud_panel = None
_hud_view = None
_hud_timer = None
_hud_history = collections.deque([0.0] * HUD_BARS, maxlen=HUD_BARS)
_level = 0.0


class HUDView(AppKit.NSView):
    def drawRect_(self, rect):
        bounds = self.bounds()
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.07, 0.07, 0.10, 0.82).set()
        AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, 18, 18
        ).fill()

        if _state == WORKING:
            self._drawDots_(bounds)
        else:
            self._drawBars_(bounds)

    def _drawBars_(self, bounds):
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 0.29, 0.33, 1.0).set()

        pad = 22.0
        usable = bounds.size.width - 2 * pad
        gap = 3.0
        width = (usable - gap * (HUD_BARS - 1)) / HUD_BARS
        mid = bounds.size.height / 2
        peak = bounds.size.height * 0.34

        for i, level in enumerate(_hud_history):
            h = max(3.0, min(1.0, level) * 2 * peak)
            x = pad + i * (width + gap)
            bar = Foundation.NSMakeRect(x, mid - h / 2, width, h)
            AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bar, width / 2, width / 2
            ).fill()

    def _drawDots_(self, bounds):
        phase = time.time() * 3.0
        radius = 5.0
        gap = 20.0
        cx = bounds.size.width / 2
        cy = bounds.size.height / 2

        for i in range(3):
            alpha = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(phase - i * 0.7))
            AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
                0.55, 0.45, 1.0, alpha
            ).set()
            x = cx + (i - 1) * gap - radius
            dot = Foundation.NSMakeRect(x, cy - radius, radius * 2, radius * 2)
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(dot).fill()


def _build_hud() -> None:
    global _hud_panel, _hud_view

    width, height = 260.0, 68.0
    frame = Foundation.NSMakeRect(0, 0, width, height)

    _hud_panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        frame,
        AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    _hud_panel.setOpaque_(False)
    _hud_panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    _hud_panel.setLevel_(AppKit.NSStatusWindowLevel)
    _hud_panel.setIgnoresMouseEvents_(True)
    _hud_panel.setHasShadow_(True)
    # Панель должна быть видна поверх всего, включая полноэкранные приложения,
    # и не таскаться между рабочими столами.
    _hud_panel.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorStationary
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
    )

    _hud_view = HUDView.alloc().initWithFrame_(frame)
    _hud_panel.setContentView_(_hud_view)


def _position_hud() -> None:
    screen = AppKit.NSScreen.mainScreen()
    if screen is None or _hud_panel is None:
        return
    visible = screen.visibleFrame()
    size = _hud_panel.frame().size
    x = visible.origin.x + (visible.size.width - size.width) / 2
    y = visible.origin.y + 120
    _hud_panel.setFrameOrigin_(Foundation.NSMakePoint(x, y))


def _hud_tick() -> None:
    _hud_history.append(_level)
    if _hud_view is not None:
        _hud_view.setNeedsDisplay_(True)


def _hud_apply(state: str) -> None:
    """Показ/скрытие панели под текущее состояние. Только главный поток."""
    global _hud_timer

    if not SHOW_HUD or _hud_panel is None:
        return

    if state in (RECORDING, WORKING):
        if state == RECORDING:
            _hud_history.extend([0.0] * HUD_BARS)
        _position_hud()
        # orderFrontRegardless, а не makeKeyAndOrderFront: панель не должна
        # забирать фокус у поля, куда потом вставится текст.
        _hud_panel.orderFrontRegardless()
        if _hud_timer is None:
            _hud_timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                1.0 / 30.0, True, lambda timer: _hud_tick()
            )
    else:
        if _hud_timer is not None:
            _hud_timer.invalidate()
            _hud_timer = None
        _hud_panel.orderOut_(None)


# ---------------------------------------------------------------------------
# Звук и уведомления
# ---------------------------------------------------------------------------


def _play_sound(name: str) -> None:
    if not SOUND_FEEDBACK:
        return
    subprocess.Popen(
        ["afplay", f"/System/Library/Sounds/{name}.aiff"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _notify(message: str) -> None:
    subprocess.Popen(
        [
            "osascript",
            "-e",
            f'display notification "{message}" with title "VoiceTyper"',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Запись звука
#
# Захват микрофона живёт в отдельном процессе (audio_helper.py). PortAudio
# после смены аудиоустройства или выхода из сна может навсегда завесить любой
# вызов в CoreAudio, причём клин процесс-глобальный: первый зависший
# Pa_StopStream делает нерабочими и все последующие Pa_OpenStream/Pa_Terminate.
# Внутри одного процесса это не лечится (проверено: watchdog с бросанием
# зависшего потока по таймауту доводил до перманентного "microphone
# unavailable" - все последующие вызовы вставали на том же клине).
# Отдельный процесс по таймауту просто убивается и поднимается заново, а всё
# записанное до клина остаётся в файле и уходит на распознавание.
# ---------------------------------------------------------------------------

HELPER_PATH = Path(__file__).resolve().with_name("audio_helper.py")

_audio_cmd: "queue.Queue[str]" = queue.Queue()
_transcribe_q: "queue.Queue[np.ndarray | None]" = queue.Queue()


class AudioHelper:
    """Клиент процесса записи: спавн, команды с таймаутом, kill при клине.

    Протокол описан в audio_helper.py. Очередь ответов создаётся заново на
    каждый спавн и передаётся читающему потоку - ответы старого, убитого
    хелпера не могут попасть новому.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._replies: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()  # спавн/kill из разных потоков

    def ensure_running(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            replies: "queue.Queue[str]" = queue.Queue()
            proc = subprocess.Popen(
                [sys.executable, str(HELPER_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            threading.Thread(
                target=self._reader,
                args=(proc, replies),
                name="helper-reader",
                daemon=True,
            ).start()
            # READY ждём до публикации процесса и строго под локом: параллельный
            # вызов не должен увидеть хелпер, который ещё не готов к командам.
            try:
                reply = replies.get(timeout=HELPER_OP_TIMEOUT)
            except queue.Empty:
                proc.kill()
                raise TimeoutError(f"хелпер молчит на старте (> {HELPER_OP_TIMEOUT:.0f}s)")
            if reply != "READY":
                proc.kill()
                raise RuntimeError(f"хелпер не поднялся: {reply!r}")
            self._proc, self._replies = proc, replies
            log.info("audio helper started (pid %d)", proc.pid)

    def kill(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2)
            log.info("audio helper killed (pid %d)", proc.pid)
        except Exception as exc:
            log.warning("не смог убить аудио-хелпер: %s", exc)

    def start(self, path: Path) -> None:
        self.ensure_running()
        self._send(f"START {path}")
        reply = self._wait_reply()
        if reply != "STARTED":
            raise RuntimeError(reply)

    def stop(self) -> None:
        self._send("STOP")
        reply = self._wait_reply()
        if not reply.startswith("STOPPED"):
            raise RuntimeError(reply)

    # -- внутренности -------------------------------------------------------

    def _send(self, line: str) -> None:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            raise RuntimeError("аудио-хелпер не запущен")
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def _wait_reply(self) -> str:
        replies = self._replies
        try:
            reply = replies.get(timeout=HELPER_OP_TIMEOUT)
        except queue.Empty:
            raise TimeoutError(f"хелпер молчит (> {HELPER_OP_TIMEOUT:.0f}s)")
        if reply.startswith("ERR"):
            raise RuntimeError(reply.partition(" ")[2] or "ошибка хелпера")
        return reply

    @staticmethod
    def _reader(proc: subprocess.Popen, replies: "queue.Queue[str]") -> None:
        """Читает stdout хелпера: LEVEL уходит в HUD, остальное - в ответы."""
        global _level
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("LEVEL "):
                try:
                    peak = float(line[6:])
                except ValueError:
                    continue
                # Быстрый подъём, плавный спад - как в старом аудио-колбэке.
                _level = max(peak * 1.6, _level * 0.55)
            else:
                replies.put(line)


_helper = AudioHelper()


def _record_path() -> Path:
    return Path(tempfile.gettempdir()) / f"voice-typer-{os.getpid()}-{time.monotonic_ns()}.raw"


def _audio_worker() -> None:
    recording: Path | None = None
    started_at = 0.0

    # Тёплый старт: процесс поднимается заранее, а не на первом нажатии Fn -
    # иначе к задержке старта записи добавился бы спавн питона (~полсекунды).
    try:
        _helper.ensure_running()
    except Exception as exc:
        log.warning("audio helper warm start failed: %s", exc)

    while True:
        try:
            cmd = _audio_cmd.get(timeout=0.5)
        except queue.Empty:
            if recording is not None and time.time() - started_at > MAX_RECORDING_SECONDS:
                log.warning("recording exceeded %.0fs, force stop", MAX_RECORDING_SECONDS)
                cmd = "stop"
            else:
                continue

        if cmd == "start":
            if recording is not None:
                continue
            path = _record_path()
            try:
                _helper.start(path)
            except Exception as exc:
                log.warning("cannot start recording: %s - перезапускаю хелпер", exc)
                _helper.kill()
                try:
                    _helper.start(path)
                except Exception as exc2:
                    log.error("microphone unavailable: %s", exc2)
                    path.unlink(missing_ok=True)
                    set_status(BROKEN, "нет доступа к микрофону")
                    _notify("Нет доступа к микрофону - проверь настройки приватности")
                    continue
            recording = path
            started_at = time.time()
            set_status(RECORDING, "запись…")
            _play_sound("Pop")
            log.info("recording started")

        elif cmd == "stop":
            if recording is None:
                continue
            try:
                _helper.stop()
            except Exception as exc:
                # Хелпер завис в CoreAudio. Убиваем его и поднимаем новый в
                # фоне; всё записанное до клина уже лежит в файле.
                log.error("остановка записи зависла (%s) - убиваю аудио-хелпер", exc)
                _helper.kill()
                threading.Thread(
                    target=_respawn_helper, name="helper-respawn", daemon=True
                ).start()
            path, recording = recording, None
            _play_sound("Tink")

            try:
                audio = np.fromfile(path, dtype=np.int16)
            except Exception as exc:
                log.error("не смог прочитать записанный звук: %s", exc)
                set_status(IDLE, "готов")
                continue
            finally:
                path.unlink(missing_ok=True)

            if audio.size == 0:
                log.info("no audio captured")
                set_status(IDLE, "готов")
                continue

            duration = audio.size / SAMPLE_RATE
            if duration < MIN_RECORDING_SECONDS:
                log.info("recording too short (%.2fs), skipping", duration)
                set_status(IDLE, "готов")
                continue

            log.info("recorded %.2fs", duration)
            # Сразу в "распознаю", не заходя в "готов": иначе панель на кадр
            # схлопывается и снова появляется.
            set_status(WORKING, "распознаю…")
            _transcribe_q.put(audio)


def _respawn_helper() -> None:
    try:
        _helper.ensure_running()
    except Exception as exc:
        log.warning("не смог поднять аудио-хелпер: %s", exc)


# ---------------------------------------------------------------------------
# Распознавание и вставка
# ---------------------------------------------------------------------------


_whisper = None


def _model_is_cached() -> bool:
    """Лежит ли модель в кэше huggingface (тогда сеть не нужна вовсе)."""
    if Path(MODEL).is_dir():  # вместо repo id указан локальный путь
        return True

    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        hub = Path(cache)
    else:
        home = os.environ.get("HF_HOME")
        hub = (Path(home) if home else Path.home() / ".cache" / "huggingface") / "hub"

    snapshots = hub / f"models--{MODEL.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    return any((snap / "config.json").exists() for snap in snapshots.iterdir())


def _load_whisper():
    """Импорт тяжёлый, поэтому ленивый и закэшированный.

    Здесь же решается, ходить ли в сеть. mlx_whisper на каждой загрузке модели
    зовёт snapshot_download, а тот - repo_info к huggingface.co, БЕЗ таймаута:
    сверяет, не появилось ли новой ревизии. Если сеть недоступна или тормозит
    (VPN, чужой Wi-Fi, captive portal), распознавание встаёт на TCP-коннекте
    до системного таймаута - минуту с лишним, молча. Инструмент офлайновый,
    модель уже в кэше - сверять нечего, поэтому сеть отключаем совсем.

    Ошибку наружу не глотаем: вызывающий её залогирует и повторит на следующей
    фразе. Один неудачный старт не должен выводить приложение из строя до
    перезапуска - раньше падение этого импорта убивало поток целиком, и
    приложение оставалось наполовину живым: запись идёт, текста нет.
    """
    global _whisper
    if _whisper is not None:
        return _whisper

    # ВАЖНО: huggingface_hub читает эти переменные один раз, при импорте,
    # поэтому выставлять их надо строго до импорта mlx_whisper.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if _model_is_cached():
        os.environ["HF_HUB_OFFLINE"] = "1"
        log.info("модель есть в кэше - работаем без сети")
    else:
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
        log.info("модели нет в кэше - будет скачана: %s (нужна сеть)", MODEL)

    import mlx_whisper

    _whisper = mlx_whisper
    return _whisper


def _transcribe_worker() -> None:
    while True:
        audio = _transcribe_q.get()
        try:
            mlx_whisper = _load_whisper()

            if audio is None:  # прогрев модели
                silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
                mlx_whisper.transcribe(
                    silence, path_or_hf_repo=MODEL, language=LANGUAGE
                )
                log.info("model ready")
            else:
                set_status(WORKING, "распознаю…")
                _transcribe_and_paste(mlx_whisper, audio)
        except Exception as exc:
            log.exception("распознавание не удалось: %s", exc)
            set_status(BROKEN, "ошибка распознавания")
            if audio is not None:
                _notify("Не удалось распознать - подробности в логе")
        else:
            set_status(IDLE, "готов")


def _transcribe_and_paste(mlx_whisper, audio: np.ndarray) -> None:
    # Массив передаётся в mlx_whisper напрямую, а не через временный .wav.
    # Путь к файлу он читает через внешний ffmpeg - которого в PATH приложения,
    # запущенного из Finder, нет (там голый /usr/bin:/bin:/usr/sbin:/sbin,
    # без /opt/homebrew/bin). Массив же уходит в mel-спектрограмму как есть:
    # ни внешних бинарников, ни файлов в /tmp. Формат тот же, что ждёт whisper:
    # моно, 16 кГц, float32 в диапазоне [-1, 1].
    #
    # reshape(-1) оставлен защитой на случай формы (N, 1): сейчас звук приходит
    # одномерным из raw-файла хелпера, но mlx падает на двумерном массиве
    # ("Shape dimension ... outside the supported range"), а страховка бесплатна.
    samples = audio.reshape(-1).astype(np.float32) / 32768.0

    t0 = time.time()
    result = mlx_whisper.transcribe(
        samples,
        path_or_hf_repo=MODEL,
        language=LANGUAGE,
        condition_on_previous_text=False,
    )
    text = result["text"].strip()
    log.info("transcribed in %.2fs: %r", time.time() - t0, text)

    if not text:
        return

    # Whisper на тишине/шуме выдаёт заученные фразы из субтитров - отсекаем.
    if text.lower().rstrip(".!… ") in HALLUCINATIONS:
        log.info("looks like a hallucination on silence, skipping")
        return

    _paste_text(text)


def _paste_text(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    time.sleep(0.15)

    # Cmd+V шлём физическим кодом клавиши (kVK_ANSI_V = 9), а не символом "v".
    # AppleScript-вариант `keystroke "v"` ищет клавишу по символу в ТЕКУЩЕЙ
    # раскладке: при русской раскладке символа "v" в ней нет, и нажатие молча
    # уходит в никуда (osascript при этом рапортует успех). Код клавиши -
    # физическая позиция, раскладка на него не влияет.
    try:
        source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        for is_down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(source, V_KEYCODE, is_down)
            Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(0.01)
        log.info("pasted")
    except Exception as exc:
        log.error("paste failed: %s", exc)
        _notify("Не удалось вставить - текст в буфере")


# ---------------------------------------------------------------------------
# Перехват Fn
# ---------------------------------------------------------------------------

_tap = None
_tap_source = None
_fn_down = False


def _fn_event_callback(proxy, event_type, event, refcon):
    """ВАЖНО: должен отрабатывать мгновенно.

    macOS отключает перехват событий, если callback тормозит
    (kCGEventTapDisabledByTimeout) - и хоткей просто перестаёт работать.
    Поэтому здесь только определяем фронт и кладём команду в очередь.
    """
    global _fn_down

    if event_type in (
        Quartz.kCGEventTapDisabledByTimeout,
        Quartz.kCGEventTapDisabledByUserInput,
    ):
        log.warning("event tap disabled by the system, re-enabling")
        Quartz.CGEventTapEnable(_tap, True)
        return event

    if event_type == Quartz.kCGEventFlagsChanged:
        pressed = bool(
            Quartz.CGEventGetFlags(event) & Quartz.kCGEventFlagMaskSecondaryFn
        )
        if pressed != _fn_down:
            _fn_down = pressed
            _audio_cmd.put("start" if pressed else "stop")

    return event


def install_tap() -> bool:
    global _tap, _tap_source

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,
        Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged),
        _fn_event_callback,
        None,
    )
    if tap is None:
        return False

    source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    Quartz.CFRunLoopAddSource(
        Quartz.CFRunLoopGetMain(), source, Quartz.kCFRunLoopCommonModes
    )
    Quartz.CGEventTapEnable(tap, True)

    # Ссылки держим глобально: если их соберёт GC, перехват тихо умрёт.
    _tap, _tap_source = tap, source
    return True


def reinstall_tap() -> None:
    global _tap, _tap_source, _fn_down

    log.warning("re-creating event tap")
    if _tap_source is not None:
        Quartz.CFRunLoopRemoveSource(
            Quartz.CFRunLoopGetMain(), _tap_source, Quartz.kCFRunLoopCommonModes
        )
    if _tap is not None:
        Quartz.CFMachPortInvalidate(_tap)
    _tap, _tap_source = None, None

    # Состояние клавиши после пересоздания неизвестно - сбрасываем в "отпущена",
    # иначе первое же нажатие будет воспринято как отпускание.
    _fn_down = False
    _audio_cmd.put("stop")

    if install_tap():
        log.info("event tap re-created")
        set_status(IDLE, "готов")
    else:
        log.error("failed to re-create event tap")
        set_status(BROKEN, "нет доступа к клавиатуре")


def _tap_watchdog() -> None:
    """Сторож перехвата.

    Главная причина "работало и перестало": macOS отключает event tap (таймаут
    обработчика, выход из сна, смена сессии), а уведомление об этом до нас не
    всегда доходит - обработчик к тому моменту уже не вызывается. Поэтому
    состояние надо опрашивать, а не ждать события.
    """
    no_permission = False
    failures = 0

    while True:
        time.sleep(TAP_CHECK_INTERVAL * (8 if failures > 3 else 1))

        # Без права "Мониторинг ввода" перехват создаётся, но остаётся выключенным
        # навсегда. Пересоздавать его в цикле бессмысленно - только ждать выдачи.
        if not Quartz.CGPreflightListenEventAccess():
            if not no_permission:
                log.error("нет права «Мониторинг ввода» - Fn не ловится")
                set_status(BROKEN, "нужен «Мониторинг ввода»")
                no_permission = True
            continue

        if no_permission:
            log.info("«Мониторинг ввода» выдан, поднимаю перехват")
            no_permission = False
            failures = 0
            reinstall_tap()
            continue

        try:
            alive = _tap is not None and Quartz.CGEventTapIsEnabled(_tap)
        except Exception as exc:
            log.warning("tap check failed: %s", exc)
            alive = False

        if alive:
            failures = 0
            continue

        if _tap is not None:
            log.warning("event tap is disabled, re-enabling")
            Quartz.CGEventTapEnable(_tap, True)
            if Quartz.CGEventTapIsEnabled(_tap):
                log.info("event tap re-enabled")
                failures = 0
                continue

        failures += 1
        if failures == 4:
            log.error("перехват не поднимается, снижаю частоту попыток")
        reinstall_tap()


# ---------------------------------------------------------------------------
# Озвучка выделенного текста (TTS)
#
# Выделил текст, нажал ⌃S - текст читается вслух голосом Silero. Повторное
# нажатие во время чтения останавливает его. Распознавание по Fn этот блок не
# трогает: свой перехват клавиш, свой поток, свой процесс-хелпер.
#
# Синтез и воспроизведение живут в отдельном процессе (tts_helper.py) - по той
# же причине, что и запись: зависший в CoreAudio вывод звука лечится только
# убийством процесса. Протокол описан в tts_helper.py.
# ---------------------------------------------------------------------------

TTS_HELPER_PATH = Path(__file__).resolve().with_name("tts_helper.py")
C_KEYCODE = 8  # kVK_ANSI_C - для копирования выделения через ⌘C

_tts_cmd: "queue.Queue[str]" = queue.Queue()


class TTSHelper:
    """Клиент процесса озвучки: спавн, команды с таймаутом, kill при клине.

    Команды шлёт только поток _tts_worker. Читающий поток разбирает ответы:
    OK/ERR - в очередь ответов, DONE <id> - конец чтения (асинхронно),
    LOG - в лог. Номер чтения нужен, чтобы DONE от прерванного чтения не
    погасил индикацию следующего.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._replies: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0
        self._active_id: int | None = None

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._active_id is not None

    def ensure_running(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        replies: "queue.Queue[str]" = queue.Queue()
        proc = subprocess.Popen(
            [
                sys.executable,
                str(TTS_HELPER_PATH),
                str(TTS_MODEL_PATH),
                TTS_MODEL_URL,
                TTS_SPEAKER,
                str(TTS_SAMPLE_RATE),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        threading.Thread(
            target=self._reader, args=(proc, replies), name="tts-reader", daemon=True
        ).start()
        if not TTS_MODEL_PATH.is_file():
            log.info("модели озвучки нет - скачиваю %s (нужна сеть)", TTS_MODEL_URL)
        try:
            reply = replies.get(timeout=TTS_START_TIMEOUT)
        except queue.Empty:
            proc.kill()
            raise TimeoutError(f"хелпер озвучки молчит на старте (> {TTS_START_TIMEOUT:.0f}s)")
        if reply != "READY":
            proc.kill()
            raise RuntimeError(reply.partition(" ")[2] or f"хелпер озвучки не поднялся: {reply!r}")
        self._proc, self._replies = proc, replies
        log.info("tts helper started (pid %d, голос %s)", proc.pid, TTS_SPEAKER)

    def speak(self, text: str) -> None:
        self.ensure_running()
        with self._lock:
            self._next_id += 1
            speak_id = self._active_id = self._next_id
        # json.dumps с ensure_ascii: переводы строк и любые символы текста
        # гарантированно уезжают одной строкой протокола.
        self._send(f"SPEAK {speak_id} {json.dumps(text)}")
        self._wait_ok()

    def stop(self) -> None:
        with self._lock:
            self._active_id = None
        if self._proc is None or self._proc.poll() is not None:
            return
        self._send("STOP")
        self._wait_ok()

    def kill(self) -> None:
        proc, self._proc = self._proc, None
        with self._lock:
            self._active_id = None
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2)
            log.info("tts helper killed (pid %d)", proc.pid)
        except Exception as exc:
            log.warning("не смог убить хелпер озвучки: %s", exc)

    # -- внутренности -------------------------------------------------------

    def _send(self, line: str) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise RuntimeError("хелпер озвучки не запущен")
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def _wait_ok(self) -> None:
        try:
            reply = self._replies.get(timeout=HELPER_OP_TIMEOUT)
        except queue.Empty:
            raise TimeoutError(f"хелпер озвучки молчит (> {HELPER_OP_TIMEOUT:.0f}s)")
        if reply != "OK":
            raise RuntimeError(reply.partition(" ")[2] or reply)

    def _reader(self, proc: subprocess.Popen, replies: "queue.Queue[str]") -> None:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("DONE "):
                try:
                    done_id = int(line[5:])
                except ValueError:
                    continue
                with self._lock:
                    finished = self._active_id == done_id
                    if finished:
                        self._active_id = None
                if finished:
                    log.info("чтение закончено")
                    _tts_finished()
            elif line.startswith("LOG "):
                log.info("tts: %s", line[4:])
            else:
                replies.put(line)
        # EOF: хелпер умер. Если он что-то читал - индикацию надо погасить.
        replies.put("ERR хелпер озвучки завершился")
        with self._lock:
            was_speaking = self._active_id is not None and self._proc is proc
            if was_speaking:
                self._active_id = None
        if was_speaking:
            log.warning("хелпер озвучки умер посреди чтения")
            _tts_finished()


_tts = TTSHelper()


def _tts_finished() -> None:
    # Гасим только свою индикацию: если за время чтения пользователь начал
    # диктовать, иконкой уже владеет распознавание.
    if _state == SPEAKING:
        set_status(IDLE, "готов")


# -- Выделенный текст ---------------------------------------------------------


def _frontmost_app() -> tuple[int, str]:
    app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return 0, "?"
    return int(app.processIdentifier()), str(app.localizedName() or app.bundleIdentifier() or "?")


def _selected_text_ax(pid: int) -> str:
    """Выделение через Accessibility API: буфер обмена не трогается.

    Работает в нативных приложениях (Safari, Заметки, Почта, Pages...).
    Electron-приложения (VS Code, Slack, Obsidian...) отдают дерево
    доступности только после AXManualAccessibility = true - выставляем его
    (для остальных приложений это просто ошибка, вреда нет). Кто не отдал
    выделение и так - пустая строка и запасной путь через ⌘C.
    """
    app = AXUIElementCreateApplication(pid)
    # Без таймаута подвисшее приложение задержало бы ответ на секунды.
    AXUIElementSetMessagingTimeout(app, 0.5)
    AXUIElementSetAttributeValue(app, "AXManualAccessibility", True)
    err, focused = AXUIElementCopyAttributeValue(app, kAXFocusedUIElementAttribute, None)
    if err != kAXErrorSuccess or focused is None:
        log.info("AX: нет сфокусированного элемента (код %s)", err)
        return ""
    err, text = AXUIElementCopyAttributeValue(focused, kAXSelectedTextAttribute, None)
    if err != kAXErrorSuccess:
        log.info("AX: выделение не отдаётся (код %s)", err)
        return ""
    return str(text or "")


def _wait_modifiers_released(timeout: float = 1.0) -> None:
    """Ждём, пока отпустят ⌃ хоткея: некоторые приложения смотрят не на флаги
    события, а на реально зажатые клавиши, и наш ⌘C превратился бы в ⌃⌘C."""
    deadline = time.time() + timeout
    mask = (
        Quartz.kCGEventFlagMaskControl
        | Quartz.kCGEventFlagMaskAlternate
        | Quartz.kCGEventFlagMaskShift
        | Quartz.kCGEventFlagMaskCommand
    )
    while time.time() < deadline:
        flags = Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateHIDSystemState)
        if not flags & mask:
            return
        time.sleep(0.02)


def _post_cmd_c() -> None:
    _wait_modifiers_released()
    # Собственный (private) источник событий: его флаги не смешиваются с
    # состоянием физической клавиатуры, ⌘C уходит ровно как ⌘C.
    source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStatePrivate)
    for is_down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(source, C_KEYCODE, is_down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.01)


def _find_copy_menu_item(element, depth: int = 0):
    """Пункт меню с хоткеем ⌘C (по хоткею, а не по названию - оно локализовано)."""
    err, kids = AXUIElementCopyAttributeValue(element, "AXChildren", None)
    if err != kAXErrorSuccess or not kids:
        return None
    for kid in kids:
        _, char = AXUIElementCopyAttributeValue(kid, "AXMenuItemCmdChar", None)
        _, mods = AXUIElementCopyAttributeValue(kid, "AXMenuItemCmdModifiers", None)
        if char and str(char).upper() == "C" and mods == 0:
            return kid
        if depth < 2:
            found = _find_copy_menu_item(kid, depth + 1)
            if found is not None:
                return found
    return None


def _press_copy_menu(pid: int) -> None:
    """Пункт «Копировать» в меню приложения, через Accessibility. Срабатывает
    там, где синтетический ⌘C игнорируется (Warp), - меню приложения доступно
    через AX, даже когда содержимое окна - нет."""
    app = AXUIElementCreateApplication(pid)
    AXUIElementSetMessagingTimeout(app, 0.5)
    err, bar = AXUIElementCopyAttributeValue(app, "AXMenuBar", None)
    if err != kAXErrorSuccess or bar is None:
        log.info("меню приложения недоступно (код %s)", err)
        return
    item = _find_copy_menu_item(bar)
    if item is None:
        log.info("в меню приложения нет пункта с ⌘C")
        return
    err = AXUIElementPerformAction(item, "AXPress")
    if err != kAXErrorSuccess:
        log.info("пункт меню ⌘C не нажался (код %s)", err)


def _selected_text_copy(trigger=_post_cmd_c) -> str:
    """Запасной путь: копирование (⌘C или пункт меню), чтение буфера и
    возврат прежнего содержимого."""
    pb = AppKit.NSPasteboard.generalPasteboard()

    saved = []
    for item in pb.pasteboardItems() or []:
        data = {}
        for kind in item.types():
            value = item.dataForType_(kind)
            if value is not None:
                data[kind] = value
        saved.append(data)
    before = pb.changeCount()

    trigger()

    # Приложение копирует асинхронно - ждём смены буфера, но недолго:
    # если ничего не выделено, буфер не изменится вовсе.
    deadline = time.time() + 1.0
    while pb.changeCount() == before and time.time() < deadline:
        time.sleep(0.02)
    if pb.changeCount() == before:
        return ""

    text = pb.stringForType_(AppKit.NSPasteboardTypeString) or ""

    pb.clearContents()
    items = []
    for data in saved:
        item = AppKit.NSPasteboardItem.alloc().init()
        for kind, value in data.items():
            item.setData_forType_(value, kind)
        items.append(item)
    if items:
        pb.writeObjects_(items)
    return str(text)


def _get_selected_text() -> str:
    pid, name = _frontmost_app()
    text = ""
    try:
        text = _selected_text_ax(pid)
    except Exception as exc:
        log.info("AX не отдал выделение: %s", exc)
    if text.strip():
        log.info("выделение через AX в «%s»: %d символов", name, len(text))
        return text
    text = _selected_text_copy()
    log.info("выделение через ⌘C в «%s»: %d символов", name, len(text))
    if text.strip():
        return text
    try:
        text = _selected_text_copy(lambda: _press_copy_menu(pid))
    except Exception as exc:
        log.info("копирование через меню не удалось: %s", exc)
        text = ""
    log.info("выделение через меню «Копировать» в «%s»: %d символов", name, len(text))
    if text.strip():
        return text
    return _selected_text_recent_clipboard(pid, name)


# -- Копирование при выделении ------------------------------------------------
#
# Терминалы вроде Warp не отдают выделение ни через Accessibility (своя
# отрисовка, AXSelectedText не поддерживается), ни на программный ⌘C. Зато они
# копируют выделенное в буфер сами, в момент выделения ("copy on select").
# Поэтому следим за буфером: когда и при каком активном приложении он менялся
# в последний раз. Если ⌘C ничего не дал, а буфер только что поменялся в том
# же приложении - это и есть выделение.

CLIPBOARD_POLL_INTERVAL = 0.25
CLIPBOARD_RECENT_SECONDS = 15.0

_clip_change = (0, 0.0, 0)  # (changeCount, время смены, pid активного приложения)


def _clipboard_watcher() -> None:
    global _clip_change
    pb = AppKit.NSPasteboard.generalPasteboard()
    last = pb.changeCount()
    while True:
        time.sleep(CLIPBOARD_POLL_INTERVAL)
        try:
            count = pb.changeCount()
            if count != last:
                last = count
                _clip_change = (count, time.time(), _frontmost_app()[0])
        except Exception as exc:
            log.warning("clipboard watcher: %s", exc)


def _selected_text_recent_clipboard(pid: int, name: str) -> str:
    count, changed_at, changed_pid = _clip_change
    pb = AppKit.NSPasteboard.generalPasteboard()
    age = time.time() - changed_at
    if count != pb.changeCount() or changed_pid != pid or age > CLIPBOARD_RECENT_SECONDS:
        log.info(
            "в буфере нет свежего выделения из «%s» (смена %.0fs назад, pid %s vs %s)",
            name, age, changed_pid, pid,
        )
        return ""
    text = str(pb.stringForType_(AppKit.NSPasteboardTypeString) or "")
    log.info("выделение из буфера (скопировано при выделении %.1fs назад): %d символов", age, len(text))
    return text


# -- Рабочий поток --------------------------------------------------------------


def _tts_worker() -> None:
    # Тёплый старт: torch и модель грузятся заранее (несколько секунд), а не
    # на первом нажатии хоткея.
    try:
        _tts.ensure_running()
    except Exception as exc:
        log.warning("хелпер озвучки не поднялся на старте: %s", exc)

    while True:
        cmd = _tts_cmd.get()
        try:
            if cmd == "stop" or (cmd == "toggle" and _tts.speaking):
                _tts.stop()
                _tts_finished()
                log.info("чтение остановлено")
                continue

            text = _get_selected_text().strip()
            if not text:
                _notify("Нечего читать - выдели текст")
                continue
            if len(text) > TTS_MAX_CHARS:
                log.info("текст длинный (%d), читаю первые %d символов", len(text), TTS_MAX_CHARS)
                text = text[:TTS_MAX_CHARS]

            set_status(SPEAKING, "читаю вслух…")
            _tts.speak(text)
            log.info("читаю вслух: %d символов", len(text))
        except Exception as exc:
            log.exception("озвучка не удалась: %s - перезапускаю хелпер", exc)
            _tts.kill()
            _tts_finished()
            _notify("Не удалось прочитать - подробности в логе")
            # Поднимаем заново сразу, а не на следующем нажатии: иначе к нему
            # добавилась бы загрузка модели. Нажатия тем временем ждут в очереди.
            try:
                _tts.ensure_running()
            except Exception as exc2:
                log.warning("не смог поднять хелпер озвучки: %s", exc2)


# -- Хоткей --------------------------------------------------------------------

_tts_tap = None
_tts_tap_source = None
_tts_swallowed_down = False


def _tts_event_callback(proxy, event_type, event, refcon):
    """Ловит ⌃S и съедает его, чтобы сочетание не ушло в активное
    приложение. Как и обработчик Fn - только классифицирует событие и кладёт
    команду в очередь: медленный обработчик macOS отключает."""
    global _tts_swallowed_down

    if event_type in (
        Quartz.kCGEventTapDisabledByTimeout,
        Quartz.kCGEventTapDisabledByUserInput,
    ):
        log.warning("tts event tap disabled by the system, re-enabling")
        Quartz.CGEventTapEnable(_tts_tap, True)
        return event

    if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode) != TTS_KEYCODE:
        return event

    if TTS_DEBUG_KEYS:
        log.info(
            "tts key: type=%s flags=%#x autorepeat=%s",
            event_type,
            Quartz.CGEventGetFlags(event),
            Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat),
        )

    if event_type == Quartz.kCGEventKeyUp:
        if _tts_swallowed_down:
            _tts_swallowed_down = False
            return None
        return event

    flags = Quartz.CGEventGetFlags(event) & (
        Quartz.kCGEventFlagMaskControl
        | Quartz.kCGEventFlagMaskAlternate
        | Quartz.kCGEventFlagMaskShift
        | Quartz.kCGEventFlagMaskCommand
    )
    if flags != TTS_MODIFIERS:
        return event

    _tts_swallowed_down = True
    if not Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat):
        _tts_cmd.put("toggle")
    return None


def install_tts_tap() -> bool:
    """Отдельный от Fn перехват: активный (может съесть событие), только
    нажатия клавиш. Активному перехвату нужно право «Универсальный доступ» -
    оно и так нужно приложению для вставки текста."""
    global _tts_tap, _tts_tap_source

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,
        Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown) | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp),
        _tts_event_callback,
        None,
    )
    if tap is None:
        return False

    source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetMain(), source, Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)
    _tts_tap, _tts_tap_source = tap, source
    return True


def _secure_input_owner() -> int:
    """PID приложения, включившего защищённый ввод (Secure Event Input), или 0.

    Пока он включён, macOS не показывает перехватчикам нажатия клавиш - ⌃S
    до нас не доходит вовсе. Изменения модификаторов при этом видны, поэтому
    Fn продолжает работать. Включают его поля паролей (браузер, менеджер
    паролей) и терминалы, когда видят запрос пароля (sudo, ssh).
    """
    info = Quartz.CGSessionCopyCurrentDictionary() or {}
    return int(info.get("kCGSSessionSecureInputPID", 0) or 0)


def _app_name(pid: int) -> str:
    app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    return str(app.localizedName()) if app is not None else f"pid {pid}"


def _tts_tap_watchdog() -> None:
    """Сторож перехвата ⌃S - по той же логике, что и сторож Fn: macOS
    отключает перехват молча, состояние надо опрашивать. Пока нет права
    «Универсальный доступ», перехват не создаётся вовсе - пробуем снова.
    Заодно следит за защищённым вводом: при нём ⌃S не ловится в принципе."""
    warned = False
    secure_pid = 0
    while True:
        time.sleep(TAP_CHECK_INTERVAL)
        try:
            pid = _secure_input_owner()
            if pid != secure_pid:
                if pid:
                    log.warning("защищённый ввод включил «%s» - %s не ловится", _app_name(pid), TTS_HOTKEY_LABEL)
                    if _state == IDLE:
                        set_status(IDLE, f"{TTS_HOTKEY_LABEL} не работает: защищённый ввод в «{_app_name(pid)}»")
                else:
                    log.info("защищённый ввод выключен - %s снова ловится", TTS_HOTKEY_LABEL)
                    if _state == IDLE:
                        set_status(IDLE, "готов")
                secure_pid = pid
        except Exception as exc:
            log.warning("secure input check failed: %s", exc)
        try:
            if _tts_tap is None:
                if install_tts_tap():
                    log.info("перехват %s установлен", TTS_HOTKEY_LABEL)
                    warned = False
                elif not warned:
                    log.error("перехват %s не создаётся - нужен «Универсальный доступ»", TTS_HOTKEY_LABEL)
                    warned = True
                continue
            if not Quartz.CGEventTapIsEnabled(_tts_tap):
                log.warning("перехват %s выключен, включаю", TTS_HOTKEY_LABEL)
                Quartz.CGEventTapEnable(_tts_tap, True)
        except Exception as exc:
            log.warning("tts tap check failed: %s", exc)


# ---------------------------------------------------------------------------
# Права доступа
# ---------------------------------------------------------------------------

SETTINGS_PANES = {
    "input": "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
}


def _open_settings(pane: str) -> None:
    subprocess.Popen(["open", SETTINGS_PANES[pane]])


def _check_permissions() -> None:
    """Права выдаются бандлу приложения, а не Python.

    Поэтому у .app и у запуска из терминала они РАЗНЫЕ: разрешив что-то
    терминалу, приложению их выдавать всё равно придётся заново.
    """
    if not Quartz.CGPreflightListenEventAccess():
        log.warning("no Input Monitoring permission, requesting")
        Quartz.CGRequestListenEventAccess()

    trusted = AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
    if not trusted:
        log.warning("no Accessibility permission (нужна для вставки Cmd+V)")


_lock_handle = None


def _acquire_single_instance() -> bool:
    """Две копии на один хоткей - это дублирование текста при вставке.

    Ловушка неочевидная: .app и запуск из терминала выглядят как разные вещи,
    но Fn ловят обе.
    """
    global _lock_handle

    _lock_handle = open(Path(tempfile.gettempdir()) / "voice-typer.lock", "w")
    try:
        fcntl.flock(_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------


def main() -> None:
    _setup_logging()
    log.info("=" * 60)
    log.info("voice-typer starting (pid %d, python %s)", os.getpid(), sys.executable)

    if not _acquire_single_instance():
        log.error("уже запущена другая копия - выхожу")
        _notify("VoiceTyper уже запущен")
        return

    app = AppKit.NSApplication.sharedApplication()
    if SHOW_IN_DOCK:
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
    else:
        # Accessory: живём в статус-баре, без иконки в Dock и без главного окна.
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    _build_main_menu()
    _build_status_item()
    _build_hud()

    _check_permissions()

    install_tap()
    # Перехват создаётся и без права "Мониторинг ввода", но остаётся выключенным,
    # поэтому реальный признак работоспособности - preflight, а не результат create.
    if Quartz.CGPreflightListenEventAccess():
        log.info("event tap installed")
    else:
        log.error("нет права «Мониторинг ввода» - Fn не ловится")
        set_status(BROKEN, "нужен «Мониторинг ввода»")
        _notify("Разреши VoiceTyper «Мониторинг ввода» - и всё заработает")
        _open_settings("input")

    threading.Thread(target=_audio_worker, name="audio", daemon=True).start()
    threading.Thread(target=_transcribe_worker, name="whisper", daemon=True).start()
    threading.Thread(target=_tap_watchdog, name="watchdog", daemon=True).start()

    if TTS_ENABLED:
        # pyobjc подгружает функции Quartz лениво, и первое обращение к одной
        # и той же функции из двух потоков сразу падает (KeyError). Сторожа Fn
        # и ⌃S просыпаются одновременно - поэтому резолвим её заранее, здесь.
        Quartz.CGEventTapIsEnabled  # noqa: B018
        if install_tts_tap():
            log.info("перехват %s установлен", TTS_HOTKEY_LABEL)
        else:
            log.error("перехват %s не создан - нужен «Универсальный доступ»", TTS_HOTKEY_LABEL)
        threading.Thread(target=_tts_worker, name="tts", daemon=True).start()
        threading.Thread(target=_clipboard_watcher, name="clipboard", daemon=True).start()
        threading.Thread(target=_tts_tap_watchdog, name="tts-watchdog", daemon=True).start()

    _transcribe_q.put(None)  # прогрев модели
    log.info("running. Hold Fn to talk.")

    app.run()


if __name__ == "__main__":
    main()
