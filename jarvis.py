"""Режим «Джарвис»: распознанная фраза уходит в Claude, ответ читается голосом.

Здесь только разговор с Claude Code CLI (`claude -p`) и подготовка ответа к
синтезу. Хоткей, индикация и озвучка живут в voice_typer.py.

Почему CLI, а не API: подписка и авторизация уже настроены в Claude Code,
отдельный ключ не нужен. Отвечает всегда агент `assistant` (личный помощник,
~/.claude/agents/300-assistant.md) - со всеми его MCP, скиллами и хуками:
заметки, почта, календарь, задачник, Telegram, NotebookLM, браузер. Каждый
вопрос - отдельный процесс `claude -p`; память разговора держится на сессии
Claude Code (`--session-id` на первом вопросе, `--resume` на следующих).

Разрешения - режим auto: безопасное агент делает сам, опасное режет
классификатор Claude Code. Изменения (письмо, сообщение, задача, событие)
агент по своим правилам выполняет только после «да» - голосом, следующей
репликой в той же сессии.
"""

import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger("voice-typer")

CLAUDE_BIN = "/opt/homebrew/bin/claude"
AGENT = "assistant"
PERMISSION_MODE = "auto"
USE_CHROME = True  # Claude in Chrome: агент может открывать и читать страницы
TIMEOUT = 300.0  # агенту с инструментами бывает нужно несколько минут
SESSION_IDLE_RESET = 600.0  # после 10 минут тишины разговор начинается заново
# Рабочий каталог - домашний: агенту доступны заметки и репозитории в ~/Devel.
WORKDIR = Path.home()

# Дописывается к системному промпту агента: он остаётся собой, меняется
# только форма ответа - его зачитывают голосом.
VOICE_PROMPT = """\
Сейчас ты говоришь с пользователем голосом, как Джарвис у Тони Старка: его \
речь распознана Whisper (возможны ошибки распознавания - угадывай смысл), \
а твой ответ сразу зачитывает синтезатор речи, который понимает только \
русский текст.

Правила ответа:
- По-русски, коротко и по делу: обычно одно-три предложения. Длиннее - только \
если прямо попросили подробно.
- Только простой текст для чтения вслух: без markdown, списков, заголовков, \
таблиц, эмодзи, ссылок и кода.
- Числа, даты, время и единицы измерения пиши словами: «двадцать пять \
градусов», «в половине седьмого».
- Английские слова и названия пиши русскими буквами, как они звучат: \
«гитхаб», «пайтон». Латиницу синтезатор пропускает.
- Не знаешь - так и скажи, не выдумывай.
- Обращайся на «ты», без лишней вежливости и вступлений.
- Не пересказывай, какие инструменты вызывал - только итог.
- Перед любым изменением (отправить письмо или сообщение, создать или \
удалить задачу, событие, файл, коммит) коротко скажи, что сделаешь, и спроси \
«делаю?». Выполняй только после явного «да» в следующей реплике.
- Если просят открыть сайт или страницу - открой её в браузере.
"""


class ClaudeError(RuntimeError):
    """Ошибка, которую стоит сказать пользователю голосом (текст - по-русски)."""


class Conversation:
    """Один разговор: сессия Claude Code и текущий запрос (для отмены)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: str | None = None
        self._started = False  # сессия уже создана в Claude Code
        self._last = 0.0
        self._proc: subprocess.Popen | None = None

    def reset(self) -> None:
        with self._lock:
            self._session = None
            self._started = False

    def cancel(self) -> None:
        """Убить текущий запрос, если он идёт (перебивание)."""
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def ask(self, question: str, on_tool=None) -> str:
        """Задать вопрос агенту. on_tool() зовётся один раз, когда агент
        впервые полез в инструменты, - чтобы сказать «секунду» вместо тишины."""
        with self._lock:
            if self._session is None or time.time() - self._last > SESSION_IDLE_RESET:
                self._session, self._started = str(uuid.uuid4()), False
                log.info("jarvis: новый разговор %s", self._session)
            session, started = self._session, self._started

        cmd = [
            CLAUDE_BIN,
            "-p",
            "--agent", AGENT,
            "--permission-mode", PERMISSION_MODE,
            "--output-format", "stream-json",
            "--verbose",  # stream-json в режиме -p без него не работает
            "--append-system-prompt", VOICE_PROMPT,
            "--resume" if started else "--session-id", session,
        ]
        if USE_CHROME:
            cmd.append("--chrome")
        cmd.append(question)

        env = dict(os.environ)
        # У приложения из Finder голый PATH, а агенту нужны gws, git, nlm и пр.
        env["PATH"] = ":".join([
            "/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".local" / "bin"),
            env.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        ])

        t0 = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=WORKDIR,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            raise ClaudeError("Не нашёл Claude Code на этом маке.")
        with self._lock:
            self._proc = proc
        timer = threading.Timer(TIMEOUT, proc.kill)
        timer.start()

        result = None
        tools_seen = False
        tail: list[str] = []
        try:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    tail = (tail + [line])[-5:]
                    continue
                kind = event.get("type")
                if kind == "assistant":
                    for block in event.get("message", {}).get("content", []) or []:
                        if block.get("type") == "tool_use":
                            log.info("jarvis: инструмент %s", block.get("name"))
                            if not tools_seen and on_tool is not None:
                                tools_seen = True
                                on_tool()
                elif kind == "result":
                    result = event
            proc.wait()
        finally:
            timer.cancel()
            with self._lock:
                self._proc = None
        err = proc.stderr.read() if proc.stderr else ""

        if proc.returncode is not None and proc.returncode < 0:
            if time.time() - t0 >= TIMEOUT - 1:
                raise ClaudeError("Не успел, задача заняла слишком долго.")
            raise InterruptedError("запрос отменён")

        if result is None:
            log.warning("jarvis: claude rc=%s без результата: %r %r", proc.returncode, tail, err[-300:])
            raise ClaudeError(_explain("".join(tail) + err))

        answer = str(result.get("result") or "")
        if result.get("is_error"):
            log.warning("jarvis: claude вернул ошибку: %r", answer[-300:])
            raise ClaudeError(_explain(answer))

        with self._lock:
            if self._session == session:
                self._started = True
                self._last = time.time()
        log.info(
            "jarvis: ответ за %.1fs (%s ходов): %r",
            time.time() - t0, result.get("num_turns"), answer[:200],
        )
        return answer


def _explain(text: str) -> str:
    low = text.lower()
    if "login" in low or "auth" in low or "api key" in low:
        return "Claude не авторизован. Войди в Claude Code в терминале."
    if "network" in low or "connect" in low or "enotfound" in low or "timed out" in low:
        return "Нет связи с Claude."
    if "rate" in low or "limit" in low:
        return "Упёрся в лимит Claude, попробуй позже."
    return "Claude ответил ошибкой, подробности в логе."


# ---------------------------------------------------------------------------
# Подготовка ответа к синтезу
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_MARKUP = re.compile(r"[*_`#>|~]+")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.M)


# -- Латиница и цифры ---------------------------------------------------------
#
# Русская модель Silero латиницу и цифры просто пропускает: «курс CS329Z»
# прозвучал бы как «курс». Агента просят писать такие слова по-русски, но он
# делает это не всегда - поэтому дочищаем сами, грубо, но слышно.

_LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи",
    "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн",
    "o": "оу", "p": "пи", "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю",
    "v": "ви", "w": "дабл ю", "x": "экс", "y": "уай", "z": "зед",
}
# Порядок важен: сначала многобуквенные сочетания.
_TRANSLIT = [
    ("tion", "шн"), ("sch", "ш"), ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"),
    ("ck", "к"), ("qu", "кв"), ("ee", "и"), ("oo", "у"), ("ea", "и"), ("ou", "ау"),
    ("ai", "эй"), ("ay", "эй"), ("oa", "оу"), ("ng", "нг"), ("kh", "х"), ("zh", "ж"),
    ("ts", "ц"), ("ya", "я"), ("yu", "ю"), ("yo", "ё"),
    ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"),
    ("h", "х"), ("i", "и"), ("j", "дж"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"),
    ("o", "о"), ("p", "п"), ("q", "к"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"),
    ("v", "в"), ("w", "в"), ("x", "кс"), ("y", "и"), ("z", "з"),
]
_SOFT_C = re.compile(r"c(?=[eiy])")


def _translit_word(word: str) -> str:
    # Аббревиатура (CS, API, GPU) читается по буквам.
    if word.isupper() and len(word) <= 5 or len(word) == 1:
        return " ".join(_LETTER_NAMES[ch] for ch in word.lower())
    w = _SOFT_C.sub("s", word.lower())
    if w.endswith("e") and len(w) > 3:  # немая e: «tune» -> «тюн», не «туне»
        w = w[:-1]
    out, i = [], 0
    while i < len(w):
        for lat, cyr in _TRANSLIT:
            if w.startswith(lat, i):
                out.append(cyr)
                i += len(lat)
                break
        else:
            i += 1
    return "".join(out)


_UNITS = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_UNITS_F = ["", "одна", "две"] + _UNITS[3:]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
          "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
         "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
             "восемьсот", "девятьсот"]
_SCALES = [("", "", "", False), ("тысяча", "тысячи", "тысяч", True),
           ("миллион", "миллиона", "миллионов", False), ("миллиард", "миллиарда", "миллиардов", False)]


def _triplet(n: int, feminine: bool) -> list[str]:
    words = [_HUNDREDS[n // 100]]
    rest = n % 100
    if 10 <= rest < 20:
        words.append(_TEENS[rest - 10])
    else:
        words.append(_TENS[rest // 10])
        words.append((_UNITS_F if feminine else _UNITS)[rest % 10])
    return [w for w in words if w]


def number_words(n: int) -> str:
    """Целое число словами, в именительном падеже: 329 -> «триста двадцать девять»."""
    if n == 0:
        return "ноль"
    if n >= 10 ** 12:
        return " ".join(_UNITS[int(d)] or "ноль" for d in str(n))  # по цифрам
    parts = []
    for one, few, many, feminine in _SCALES:
        n, chunk = divmod(n, 1000)
        if chunk:
            words = _triplet(chunk, feminine)
            last2, last = chunk % 100, chunk % 10
            if one:
                form = many if 10 <= last2 < 20 or last == 0 or last >= 5 else one if last == 1 else few
                words.append(form)
            parts.insert(0, " ".join(words))
        if not n:
            break
    text = " ".join(parts)
    return text[len("одна "):] if text.startswith("одна тысяча") else text


_TOKEN = re.compile(r"[A-Za-z]+|\d+")


def russify(text: str) -> str:
    """Латиницу - русскими буквами, цифры - словами. «CS329Z» -> «си эс триста
    двадцать девять зед»."""
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        word = number_words(int(tok)) if tok.isdigit() else _translit_word(tok)
        return f" {word} "
    text = re.sub(r"(?<=\d)[.,](?=\d)", " целых ", text)  # 2.5 -> «два целых пять»
    text = text.replace("&", " и ").replace("%", " процентов ").replace("+", " плюс ")
    text = _TOKEN.sub(repl, text)
    text = " ".join(text.split())
    text = re.sub(r"\s+([.,!?…:;»)])", r"\1", text)
    return re.sub(r"([«(])\s+", r"\1", text)


# -- Зацикленное распознавание ---------------------------------------------------

_WORD = re.compile(r"\w+")


def looks_looped(text: str) -> bool:
    """Whisper иногда зацикливается и пишет одно слово десятки раз («очень
    очень очень…»). Такую строку агенту не отдаём."""
    words = _WORD.findall(text.lower())
    if len(words) < 8:
        return False
    run, best = 1, 1
    for prev, cur in zip(words, words[1:]):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    if best >= 6:
        return True
    return len(words) >= 20 and len(set(words)) / len(words) < 0.25


def speakable(text: str) -> str:
    """Убирает разметку, которую модель всё же могла вставить: синтезатор
    прочитал бы её как мусор или споткнулся."""
    text = _CODE_BLOCK.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _BULLET.sub("", text)
    text = _MARKUP.sub(" ", text)
    # Строки списка без точки в конце склеились бы в одно предложение.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln if ln[-1] in ".!?…:;," else ln + "." for ln in lines]
    return russify(" ".join(lines))
