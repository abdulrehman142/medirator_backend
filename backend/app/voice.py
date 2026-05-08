"""Optional speech-to-text used before routing / model calls."""

from __future__ import annotations

from pathlib import Path


def transcribe_audio_file(path: str | Path) -> str:
    """
    Lightweight transcription via SpeechRecognition + Google web API.
    Raises RuntimeError if the stack is missing or audio cannot be decoded.
    """
    try:
        import speech_recognition as sr  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the `SpeechRecognition` package to enable voice input.") from exc

    audio_file = Path(path)
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(str(audio_file)) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data).strip()
    except sr.UnknownValueError as exc:
        raise RuntimeError("Speech was not understandable.") from exc
    except sr.RequestError as exc:
        raise RuntimeError("Speech recognition service unavailable.") from exc
