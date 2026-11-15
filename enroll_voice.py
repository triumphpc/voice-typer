#!/usr/bin/env python3
"""Запись отпечатка голоса для режима Джарвис (см. speaker_gate.py).

    .venv/bin/python enroll_voice.py          записать свой голос (6 фраз)
    .venv/bin/python enroll_voice.py --test   слушать и печатать оценку
                                              каждой фразы - для подбора порога

Фразы режутся тем же детектором, что и в режиме Джарвис (listen_helper.py),
так что отпечаток строится на звуке той же формы, что и при работе.
Говори обычным голосом, с того места, откуда обычно говоришь с Джарвисом.
"""

import argparse
import queue
import sys

import numpy as np
import sounddevice as sd

import speaker_gate
from listen_helper import FRAME, SAMPLE_RATE, Endpointer

PHRASES = [
    "Джарвис, что у меня сегодня в календаре?",
    "Напомни завтра в девять утра позвонить в сервис.",
    "Найди в почте последнее письмо про подписку.",
    "Сколько задач у меня просрочено на этой неделе?",
    "Расскажи коротко, какая погода будет на выходных.",
    "Запиши в заметки: купить фильтры для воды и батарейки.",
]
MIN_PHRASE_SEC = 1.5


def _phrases(stream_q: "queue.Queue[np.ndarray]"):
    """Бесконечный генератор готовых фраз (int16) из микрофона."""
    ep = Endpointer()
    buf = np.zeros(0, dtype=np.int16)
    while True:
        buf = np.concatenate([buf, stream_q.get()])
        while buf.size >= FRAME:
            frame, buf = buf[:FRAME], buf[FRAME:]
            event = ep.feed(frame)
            if isinstance(event, np.ndarray):
                yield event
            elif event == "drop":
                print("   (шум, не считаю)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="печатать оценку каждой фразы")
    args = parser.parse_args()

    gate = speaker_gate.SpeakerGate()
    print("Загружаю модель…")
    gate.load()

    stream_q: "queue.Queue[np.ndarray]" = queue.Queue()
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
        callback=lambda indata, *_: stream_q.put(indata[:, 0].copy()),
    )
    stream.start()
    phrases = _phrases(stream_q)

    if args.test:
        if not gate.has_voiceprint():
            print("Отпечатка ещё нет - сначала запусти без --test.")
            return 1
        print("Говори (или включи телевизор, дай сказать другому). Ctrl+C - выход.\n")
        try:
            for audio in phrases:
                print(f"  {audio.size / SAMPLE_RATE:4.1f} с   оценка {gate.score(audio):.3f}")
        except KeyboardInterrupt:
            return 0

    print(f"\nПрочитай вслух {len(PHRASES)} фраз, по одной. После каждой - короткая пауза.")
    print("Режим Джарвис должен быть выключен (⌃J), иначе он будет отвечать в микрофон.\n")
    embeddings, durations = [], []
    try:
        for i, text in enumerate(PHRASES, 1):
            print(f"{i}/{len(PHRASES)}  «{text}»")
            while True:
                audio = next(phrases)
                dur = audio.size / SAMPLE_RATE
                if dur < MIN_PHRASE_SEC:
                    print(f"   слишком коротко ({dur:.1f} с), ещё раз")
                    continue
                embeddings.append(gate.embed(audio))
                durations.append(dur)
                print(f"   записано, {dur:.1f} с")
                break
    except KeyboardInterrupt:
        print("\nПрервано, отпечаток не менял.")
        return 1
    stream.stop()

    speaker_gate.save_voiceprint(embeddings)
    # Каждую фразу сравниваем с отпечатком из остальных - честнее, чем с
    # отпечатком, в который она сама вошла.
    scores = []
    for i, e in enumerate(embeddings):
        others = np.mean([x for j, x in enumerate(embeddings) if j != i], axis=0)
        scores.append(float(e @ (others / np.linalg.norm(others))))
    print(f"\nОтпечаток сохранён: {speaker_gate.VOICEPRINT_PATH}")
    print("Похожесть каждой фразы на остальные: " + ", ".join(f"{s:.2f}" for s in scores))
    print(f"Минимум {min(scores):.2f}. Порог в voice_typer.py (JARVIS_VOICE_THRESHOLD) "
          "должен быть заметно ниже твоих оценок и выше чужих - проверь через --test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
