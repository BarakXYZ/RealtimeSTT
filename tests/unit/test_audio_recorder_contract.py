import importlib
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_module(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _install_dependency_stubs() -> None:
    faster_whisper = types.ModuleType("faster_whisper")

    class _FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def transcribe(self, *args, **kwargs):
            return [], types.SimpleNamespace(language="en", language_probability=1.0)

    class _FakeBatchedInferencePipeline:
        def __init__(self, model):
            self.model = model

        def transcribe(self, *args, **kwargs):
            return self.model.transcribe(*args, **kwargs)

    faster_whisper.WhisperModel = _FakeWhisperModel
    faster_whisper.BatchedInferencePipeline = _FakeBatchedInferencePipeline
    _stub_module("faster_whisper", faster_whisper)

    openwakeword = types.ModuleType("openwakeword")
    openwakeword_model = types.ModuleType("openwakeword.model")
    openwakeword_model.Model = type("Model", (), {})
    _stub_module("openwakeword", openwakeword)
    _stub_module("openwakeword.model", openwakeword_model)

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.hub = types.SimpleNamespace(load=lambda *args, **kwargs: (None, None))
    torch_multiprocessing = types.ModuleType("torch.multiprocessing")
    torch_multiprocessing.Value = lambda *args, **kwargs: types.SimpleNamespace(value=args[1] if len(args) > 1 else None)
    torch.multiprocessing = torch_multiprocessing
    _stub_module("torch", torch)
    _stub_module("torch.multiprocessing", torch_multiprocessing)

    scipy = types.ModuleType("scipy")
    scipy_signal = types.ModuleType("scipy.signal")
    scipy_signal.resample = lambda data, new_size: data
    scipy_signal.resample_poly = lambda data, up, down: data
    scipy_signal.butter = lambda *args, **kwargs: (None, None)
    scipy_signal.filtfilt = lambda b, a, data: data
    scipy.signal = scipy_signal
    _stub_module("scipy", scipy)
    _stub_module("scipy.signal", scipy_signal)

    soundfile = types.ModuleType("soundfile")
    soundfile.read = lambda *args, **kwargs: (np.zeros(160, dtype=np.float32), 16000)
    _stub_module("soundfile", soundfile)

    halo = types.ModuleType("halo")
    halo.Halo = type("Halo", (), {"__init__": lambda self, *args, **kwargs: None, "start": lambda self: None, "stop": lambda self: None})
    _stub_module("halo", halo)

    websocket = types.ModuleType("websocket")
    websocket.WebSocketApp = type("WebSocketApp", (), {})
    websocket.ABNF = types.SimpleNamespace(OPCODE_BINARY=2)
    _stub_module("websocket", websocket)

    pyaudio = types.ModuleType("pyaudio")
    pyaudio.paInt16 = 8
    pyaudio.PyAudio = type("PyAudio", (), {})
    _stub_module("pyaudio", pyaudio)

    colorama = types.ModuleType("colorama")
    colorama.init = lambda *args, **kwargs: None
    colorama.Fore = types.SimpleNamespace(LIGHTGREEN_EX="", CYAN="", YELLOW="", LIGHTBLUE_EX="")
    colorama.Style = types.SimpleNamespace(RESET_ALL="")
    _stub_module("colorama", colorama)

    webrtcvad = types.ModuleType("webrtcvad")
    webrtcvad.Vad = type("Vad", (), {"__init__": lambda self, *args, **kwargs: None, "is_speech": lambda self, *args, **kwargs: False})
    _stub_module("webrtcvad", webrtcvad)

    pvporcupine = types.ModuleType("pvporcupine")
    pvporcupine.create = lambda *args, **kwargs: types.SimpleNamespace(process=lambda pcm: -1, delete=lambda: None)
    _stub_module("pvporcupine", pvporcupine)


_install_dependency_stubs()
audio_recorder = importlib.import_module("RealtimeSTT.audio_recorder")
interfaces = importlib.import_module("RealtimeSTT.asr.interfaces")


class AudioRecorderContractTests(unittest.TestCase):
    def _make_recorder(self):
        recorder = audio_recorder.AudioToTextRecorder.__new__(audio_recorder.AudioToTextRecorder)
        recorder.transcription_lock = threading.Lock()
        recorder.parent_transcription_pipe = None
        recorder.transcribe_count = 0
        recorder.allowed_to_early_transcribe = False
        recorder.detected_language = None
        recorder.detected_language_probability = 0.0
        recorder.last_transcription_bytes = None
        recorder.last_transcription_bytes_b64 = None
        recorder.print_transcription_time = False
        recorder.main_model_type = "tiny"
        recorder.language = "en"
        recorder.ensure_sentence_starting_uppercase = True
        recorder.ensure_sentence_ends_with_period = True
        recorder.interrupt_stop_event = threading.Event()
        recorder.was_interrupted = threading.Event()
        recorder._set_state = lambda value: setattr(recorder, "_state", value)
        return recorder

    def test_final_transcription_preprocesses_and_tracks_language(self):
        recorder = self._make_recorder()
        audio = np.array([1, 2, 3], dtype=np.int16)
        info = types.SimpleNamespace(language="en", language_probability=0.91)

        class Pipe:
            def send(self, payload):
                self.payload = payload

            def poll(self, timeout):
                return True

            def recv(self):
                return ("success", ("hello world", info))

        recorder.parent_transcription_pipe = Pipe()

        with patch("RealtimeSTT.audio_recorder.time.time", side_effect=[100.0, 101.5]):
            result = recorder.perform_final_transcription(audio)

        self.assertEqual(result, "Hello world.")
        self.assertEqual(recorder.detected_language, "en")
        self.assertEqual(recorder.detected_language_probability, 0.91)
        self.assertTrue(recorder.allowed_to_early_transcribe)
        self.assertIsNotNone(recorder.last_transcription_bytes_b64)
        self.assertEqual(recorder._state, "inactive")

    def test_final_transcription_returns_empty_when_interrupted(self):
        recorder = self._make_recorder()
        audio = np.array([1, 2, 3], dtype=np.int16)

        class Pipe:
            def send(self, payload):
                self.payload = payload

            def poll(self, timeout):
                recorder.interrupt_stop_event.set()
                return False

            def recv(self):
                raise AssertionError("recv should not be called after interruption")

        recorder.parent_transcription_pipe = Pipe()

        result = recorder.perform_final_transcription(audio)

        self.assertEqual(result, "")
        self.assertTrue(recorder.was_interrupted.is_set())
        self.assertEqual(recorder._state, "inactive")

    def test_preprocess_output_keeps_preview_without_period(self):
        recorder = self._make_recorder()

        preview = recorder._preprocess_output("  hello\n world  ", preview=True)
        final = recorder._preprocess_output("  hello\n world  ", preview=False)

        self.assertEqual(preview, "Hello world")
        self.assertEqual(final, "Hello world.")

    def test_whisper_cpp_realtime_defaults_force_safe_streaming_shape(self):
        recorder = self._make_recorder()
        recorder.use_main_model_for_realtime = True
        recorder.beam_size_realtime = 3
        recorder.whisper_cpp_no_context_realtime = True
        recorder.whisper_cpp_single_segment_realtime = True
        recorder.whisper_cpp_stream_step_ms = 100
        recorder.whisper_cpp_stream_length_ms = 300
        recorder.whisper_cpp_stream_keep_ms = 900
        recorder.realtime_processing_pause = 0.03

        recorder._apply_whisper_cpp_realtime_defaults()

        self.assertFalse(recorder.use_main_model_for_realtime)
        self.assertEqual(recorder.beam_size_realtime, 1)
        self.assertFalse(recorder.whisper_cpp_no_context_realtime)
        self.assertFalse(recorder.whisper_cpp_single_segment_realtime)
        self.assertEqual(recorder.whisper_cpp_stream_step_ms, 700)
        self.assertEqual(recorder.whisper_cpp_stream_length_ms, 700)
        self.assertEqual(recorder.whisper_cpp_stream_keep_ms, 700)
        self.assertAlmostEqual(recorder.realtime_processing_pause, 0.7)

    def test_whisper_cpp_realtime_text_builder_commits_only_stable_segments(self):
        recorder = self._make_recorder()
        recorder.whisper_cpp_stream_keep_ms = 200
        recorder.realtime_stream_committed_text = ""
        recorder.realtime_stabilized_safetext = ""
        recorder.realtime_stabilized_text = ""
        recorder.realtime_transcription_text = ""
        recorder.text_storage = []

        transcript = interfaces.TranscriptResult(
            text="hello brave new world",
            segments=[
                interfaces.TranscriptSegment(text="hello brave", t0_ms=0, t1_ms=1200),
                interfaces.TranscriptSegment(text="new world", t0_ms=1200, t1_ms=4950),
            ],
            metadata=interfaces.TranscriptMetadata(
                language="en",
                language_probability=1.0,
                backend_name="whisper.cpp",
                model_id="base.en",
                timings={},
            ),
        )

        committed, preview = recorder._build_whisper_cpp_realtime_texts(transcript, window_duration_ms=5000)

        self.assertEqual(committed, "hello brave")
        self.assertEqual(preview, "hello brave new world")


if __name__ == "__main__":
    unittest.main()
