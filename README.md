# VoiceTyper

[![CI](https://github.com/triumphpc/voice-typer/actions/workflows/ci.yml/badge.svg)](https://github.com/triumphpc/voice-typer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![macOS 13+](https://img.shields.io/badge/macOS-13%2B-black?logo=apple)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-orange)
![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)

**Push-to-talk voice typing for macOS — fully offline.**

Hold **Fn**, speak, release. Your words are transcribed by a local Whisper
model ([mlx-whisper](https://github.com/ml-explore/mlx-examples), accelerated
on Apple Silicon) and pasted into whatever text field has focus.

No cloud, no subscriptions, no telemetry. Audio and text never leave your Mac.

*Документация на русском: [README.ru.md](README.ru.md).*

## Features

- 🎙 **Push-to-talk** — hold the Fn key anywhere, in any app
- 🔒 **100% offline** — transcription runs locally via Whisper on MLX;
  after the first model download the app never touches the network
- ⚡️ **Fast** — ~0.5 s to transcribe a typical phrase on an M-series chip
- 📊 **Live HUD** — a small floating panel shows the microphone level while
  recording and a "thinking" indicator while transcribing; no beeps,
  meeting-friendly
- 🍏 **Native menu bar app** — status icon, log access, quit; no Dock icon
- 🛡 **Self-healing** — watchdogs recover from macOS disabling the event tap,
  audio-device changes, sleep/wake, and wedged CoreAudio calls
- 🌍 **Any Whisper language** — defaults to Russian, one line to change
- 🔊 **Read selection aloud** — select text anywhere, press **⌃S**, and a
  local Silero voice reads it; press again to stop

## How it works

```
   hold Fn                release Fn
      │                       │
      ▼                       ▼
┌───────────┐   audio   ┌───────────┐   text   ┌──────────────┐
│ CGEventTap│──────────▶│  Whisper  │─────────▶│ clipboard +  │
│ (hotkey)  │  PortAudio│ (MLX, GPU)│          │ synthetic ⌘V │
└───────────┘           └───────────┘          └──────────────┘
```

Everything runs in a single menu bar app. The hotkey is captured with a Quartz
`CGEventTap` (Fn is a modifier flag, ordinary hotkey libraries can't see it),
audio is recorded with PortAudio in a dedicated helper process (so a wedged
CoreAudio call can be killed and respawned), transcription runs on the GPU via
MLX, and
the result is pasted with a synthetic Cmd+V keystroke sent by physical keycode
— so it works regardless of the active keyboard layout.

## Requirements

- macOS 13+ on Apple Silicon (M1 or newer) — MLX requires it
- [python.org framework build of Python 3.13](https://www.python.org/downloads/macos/)
  (`/Library/Frameworks/Python.framework`) — the app embeds this interpreter
- ~1.6 GB of disk for the default Whisper model (downloaded once on first run)

## Installation

```bash
git clone https://github.com/triumphpc/voice-typer.git
cd voice-typer
make install
```

This builds `/Applications/VoiceTyper.app`. Launch it like any app — Finder,
Launchpad, or Spotlight. To start it automatically, add VoiceTyper to
**System Settings → General → Login Items**.

> The bundle references this project directory rather than copying it:
> edits to `voice_typer.py` take effect on the next launch, no rebuild needed.
> If you move the directory, rebuild with `make install`.

### Permissions

On first launch macOS will ask for access. In **System Settings →
Privacy & Security**, grant:

| Permission | Why |
|---|---|
| **Microphone** | record your voice |
| **Input Monitoring** | detect the Fn key globally |
| **Accessibility** | paste text via Cmd+V, read the selection on ⌃S |

Permissions are granted to the **app bundle**, not to Python — anything you
previously granted to your terminal does not carry over. This is also why the
launcher embeds the interpreter into the app binary instead of `exec`-ing
`python3`: macOS ties permissions to the executable image, and a naive
launcher silently loses them.

Also set **System Settings → Keyboard → "Press Fn key to" → Do Nothing**,
otherwise Fn will simultaneously open the emoji picker or system dictation.

## Usage

1. Put the cursor in any text field
2. Hold **Fn**, say a phrase, release
3. The text appears

While recording, a floating panel above the Dock shows the live microphone
level; after you release Fn it switches to pulsing dots while Whisper works,
then hides. The menu bar icon mirrors the state: mic — ready, filled mic —
recording, waveform — transcribing, warning triangle — missing permissions.

If pasting fails, the text is still on the clipboard — press Cmd+V manually.

### Reading text aloud

Select text in any app and press **⌃S** — it is read aloud by a local
Silero voice (Russian, `aidar` by default). Press **⌃S** again, or use
"Stop reading" in the menu, to stop. The menu bar icon shows a speaker while
reading. Dictation on Fn works independently.

The selection is taken through the Accessibility API, which leaves the
clipboard alone. Apps that don't expose it (Electron, some browsers) fall
back to a simulated ⌘C, after which the previous clipboard contents are
restored. If the app ignores synthetic ⌘C too (Warp does), the app's own
Edit → Copy menu item is pressed through Accessibility. The last resort is
copy-on-select terminals: whatever the app itself put on the clipboard in
the last 15 s.

While some app holds Secure Event Input (a password field, `sudo`, the lock
screen), macOS hides key presses from every event tap, so ⌃S is not seen;
the menu status says so. Fn keeps working.

The Silero model (`v5_6_ru`, ~150 MB) is downloaded once on first launch to
`~/.cache/voice-typer/`; after that no network is used.

## Configuration

All settings are constants at the top of [`voice_typer.py`](voice_typer.py):

| Setting | Default | Description |
|---|---|---|
| `MODEL` | `whisper-large-v3-turbo` | Whisper model; `small`/`base` are faster but less accurate |
| `LANGUAGE` | `"ru"` | fixed language (faster, more accurate); `None` = auto-detect |
| `MIN_RECORDING_SECONDS` | `0.4` | recordings shorter than this are discarded |
| `MAX_RECORDING_SECONDS` | `120` | safety cap in case an Fn release event is lost |
| `SHOW_HUD` / `HUD_BARS` | `True` / `27` | the floating level panel |
| `SOUND_FEEDBACK` | `False` | start/stop sounds (off — the HUD replaces them) |
| `SHOW_IN_DOCK` | `False` | show a Dock icon and appear in ⌘-Tab |
| `HALLUCINATIONS` | — | phrases Whisper invents on silence, filtered out |
| `TTS_ENABLED` | `True` | reading the selection aloud on ⌃S |
| `TTS_SPEAKER` | `"aidar"` | Silero voice: `aidar`, `eugene` (male), `baya`, `kseniya`, `xenia` (female) |
| `TTS_SAMPLE_RATE` | `48000` | synthesis sample rate: `8000` / `24000` / `48000` |
| `TTS_KEYCODE` / `TTS_MODIFIERS` | S / ⌃ | the read-aloud hotkey (physical key code, layout-independent) |
| `TTS_MAX_CHARS` | `20000` | longer selections are truncated |

## Troubleshooting

**Log:** `~/Library/Logs/VoiceTyper.log` (or "Show log" in the menu). Every
step is logged: recording start, duration, transcription time, result,
permission and event-tap issues.

**Debug run** in a terminal with console output:

```bash
make run
```

Only one instance runs at a time — the console copy won't start while the app
is running (otherwise both would capture Fn and paste twice).

**If the app hangs** (icon present, no reaction) — dump all thread stacks and
see exactly where it is stuck:

```bash
kill -USR1 $(pgrep -f VoiceTyper.app)
cat ~/Library/Logs/VoiceTyper.threads.log
```

**Model updates** are manual by design — the app deliberately never checks
huggingface.co for new revisions (an offline tool must not block on the
network). To update, delete the model from
`~/.cache/huggingface/hub/models--mlx-community--*` and launch with network
access.

## Design notes

The app is built around one observation: on macOS, every failure mode of a
background utility tends to be **silent**. Each of these was hit in practice
and is now handled:

- **Network stalls.** `mlx_whisper` calls `snapshot_download` on every model
  load, which pings huggingface.co *without a timeout*. On a flaky network
  (VPN, captive portal) transcription silently blocked for over a minute.
  Now, if the model is cached, `HF_HUB_OFFLINE=1` is set before import and
  the network is never used.
- **Event tap death.** macOS disables event taps (handler timeout, sleep/wake,
  session switch) and the notification often never arrives. A watchdog thread
  polls the tap every 2 s and re-enables or re-creates it.
- **Wedged CoreAudio.** `Pa_StopStream` can block forever after an
  audio-device change or wake from sleep — and the wedge is process-global:
  once one call is stuck, every later `Pa_OpenStream`/`Pa_Terminate` hangs
  too, so no in-process timeout can recover (tried; it ended in a permanent
  "microphone unavailable"). Audio capture therefore lives in a separate
  helper process (`audio_helper.py`): commands go over stdin/stdout with a
  timeout, frames are written unbuffered to a file, and a wedged helper is
  simply killed and respawned — everything captured before the hang is
  still transcribed.
- **Layout-proof pasting.** Cmd+V is posted by physical keycode
  (`kVK_ANSI_V`), not by character — AppleScript's `keystroke "v"` looks the
  key up in the *current* layout and silently does nothing on non-Latin
  layouts.
- **No external binaries.** Audio is passed to the model as an in-memory
  array, not a temp `.wav` — apps launched from Finder don't have Homebrew's
  `ffmpeg` in `PATH`.

Worker-thread crashes are logged via `threading.excepthook`, the transcription
thread survives errors and retries on the next phrase, and the menu bar icon
always reflects the real state.

## Known limitations

- macOS on Apple Silicon only (MLX + Accessibility APIs)
- Pasting goes through the clipboard, overwriting its previous contents
- On very quiet or short recordings Whisper occasionally hallucinates text —
  common cases are filtered, the rest is one Cmd+Z away
- Read-aloud is Russian only: Silero's Russian model skips Latin words and
  digits (write numbers out as words to hear them)

## License

[MIT](LICENSE)
