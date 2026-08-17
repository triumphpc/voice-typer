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
import sounddevice as sd

import AppKit
import Foundation
import Quartz
from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
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

# Потолок на любой вызов PortAudio (открытие/остановка потока, переинициализация).
# Pa_StopStream после смены аудиоустройства или выхода из сна может заблокироваться
# в CoreAudio навсегда - без таймаута зависает весь аудио-поток, и хоткей умирает.
PORTAUDIO_OP_TIMEOUT = 5.0

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

# Иконка в строке меню: SF Symbol, а не эмодзи. Эмодзи в строке меню рисуется
# цветным глифом непредсказуемой ширины и плохо читается на тёмном фоне;
# template-изображение macOS перекрашивает под тему сама.
MENUBAR_SYMBOLS = {
    IDLE: ("mic", "🎙"),
    RECORDING: ("mic.fill", "🔴"),
    WORKING: ("waveform", "…"),
    BROKEN: ("exclamationmark.triangle", "⚠️"),
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

    menu.addItem_(AppKit.NSMenuItem.separatorItem())
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
# Запуск и остановка потока PortAudio идут в одном выделенном треде, командами
# через очередь. Раньше это делалось прямо из обработчика хоткея под общим
# мьютексом - и любой залипший вызов PortAudio (смена аудиоустройства, выход из
# сна) оставлял мьютекс захваченным навсегда: процесс жив, а хоткей больше
# ничего не делает. Очередь такого класса отказов не допускает.
# ---------------------------------------------------------------------------

_audio_cmd: "queue.Queue[str]" = queue.Queue()
_transcribe_q: "queue.Queue[np.ndarray | None]" = queue.Queue()


def _pa_call(fn, name: str):
    """Выполняет вызов PortAudio в отдельном потоке с таймаутом.

    Любой вызов PortAudio (Pa_StopStream, Pa_OpenStream, Pa_Terminate) может
    заблокироваться в CoreAudio навсегда после смены устройства или сна.
    Если это случится в аудио-потоке напрямую - очередь команд встанет, и
    приложение перестанет реагировать на Fn при живом процессе (наблюдалось
    вживую: стек показывал вечный Pa_StopStream). По таймауту зависший вызов
    бросаем: его поток остаётся висеть в C-коде (утечка одного потока), зато
    аудио-воркер продолжает работать.
    """
    result: dict = {}
    done = threading.Event()

    def run():
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - пробрасываем вызывающему
            result["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, name=name, daemon=True).start()
    if not done.wait(PORTAUDIO_OP_TIMEOUT):
        raise TimeoutError(f"{name} завис (> {PORTAUDIO_OP_TIMEOUT:.0f}s)")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _make_audio_callback(frames: list):
    """Колбэк с собственным буфером кадров.

    Буфер обязан принадлежать конкретному потоку записи: если stop() завис и
    стрим брошен, его колбэк может продолжать дёргаться - и с общим глобальным
    буфером он подмешивал бы старый звук в следующую запись.
    """

    def callback(indata, frame_count, time_info, status):
        global _level
        if status:
            log.debug("audio status: %s", status)
        frames.append(indata.copy())

        # Уровень для панели. Считается здесь же, потому что это единственное
        # место, где звук вообще есть; стоит копейки и не задерживает PortAudio.
        peak = float(np.abs(indata).max()) / 32768.0
        _level = max(peak * 1.6, _level * 0.55)  # быстрый подъём, плавный спад

    return callback


def _reset_portaudio() -> None:
    """PortAudio держит снимок аудиоустройств с момента инициализации.

    После смены устройства (наушники, док, выход из сна) старый снимок
    протухает и открытие потока падает раз за разом. Переинициализация
    заставляет его перечитать устройства.
    """
    try:
        _pa_call(lambda: (sd._terminate(), sd._initialize()), "pa-reset")
        log.info("portaudio re-initialised")
    except Exception as exc:
        log.warning("portaudio re-init failed: %s", exc)


def _open_stream():
    frames: list = []

    def do_open():
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=_make_audio_callback(frames),
        )
        stream.start()
        return stream

    return _pa_call(do_open, "pa-open"), frames


def _close_stream(stream) -> None:
    def do_close():
        stream.stop()
        stream.close()

    try:
        _pa_call(do_close, "pa-close")
    except TimeoutError as exc:
        # Стрим бросаем как есть: его поток завис в CoreAudio. Уже записанные
        # кадры при этом целы - их можно распознавать.
        log.error("остановка записи зависла, бросаю аудиопоток: %s", exc)
    except Exception as exc:
        log.warning("error while closing stream: %s", exc)
        _reset_portaudio()


def _audio_worker() -> None:
    stream = None
    frames: list = []
    started_at = 0.0

    while True:
        try:
            cmd = _audio_cmd.get(timeout=0.5)
        except queue.Empty:
            if stream is not None and time.time() - started_at > MAX_RECORDING_SECONDS:
                log.warning("recording exceeded %.0fs, force stop", MAX_RECORDING_SECONDS)
                cmd = "stop"
            else:
                continue

        if cmd == "start":
            if stream is not None:
                continue
            try:
                stream, frames = _open_stream()
            except Exception as exc:
                log.warning("cannot open microphone: %s - retrying after reset", exc)
                _reset_portaudio()
                try:
                    stream, frames = _open_stream()
                except Exception as exc2:
                    stream = None
                    log.error("microphone unavailable: %s", exc2)
                    set_status(BROKEN, "нет доступа к микрофону")
                    _notify("Нет доступа к микрофону - проверь настройки приватности")
                    continue
            started_at = time.time()
            set_status(RECORDING, "запись…")
            _play_sound("Pop")
            log.info("recording started")

        elif cmd == "stop":
            if stream is None:
                continue
            _close_stream(stream)
            stream = None

            captured, frames = frames, []
            _play_sound("Tink")

            if not captured:
                log.info("no audio captured")
                set_status(IDLE, "готов")
                continue

            audio = np.concatenate(captured, axis=0)
            duration = len(audio) / SAMPLE_RATE
            if duration < MIN_RECORDING_SECONDS:
                log.info("recording too short (%.2fs), skipping", duration)
                set_status(IDLE, "готов")
                continue

            log.info("recorded %.2fs", duration)
            # Сразу в "распознаю", не заходя в "готов": иначе панель на кадр
            # схлопывается и снова появляется.
            set_status(WORKING, "распознаю…")
            _transcribe_q.put(audio)


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
    # reshape обязателен: sounddevice отдаёт кадры как (N, каналы), то есть
    # (N, 1) даже для моно. Раньше форму выпрямляла запись в wav; если отдать
    # двумерный массив как есть, mlx падает на подсчёте окон БПФ
    # ("Shape dimension ... outside the supported range").
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

    _transcribe_q.put(None)  # прогрев модели
    log.info("running. Hold Fn to talk.")

    app.run()


if __name__ == "__main__":
    main()
