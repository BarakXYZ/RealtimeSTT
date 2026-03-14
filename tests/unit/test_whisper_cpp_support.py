import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_module(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _install_client_dependency_stubs() -> None:
    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = type("WhisperModel", (), {})
    faster_whisper.BatchedInferencePipeline = type("BatchedInferencePipeline", (), {})
    _stub_module("faster_whisper", faster_whisper)

    openwakeword = types.ModuleType("openwakeword")
    openwakeword_model = types.ModuleType("openwakeword.model")
    openwakeword_model.Model = type("Model", (), {})
    _stub_module("openwakeword", openwakeword)
    _stub_module("openwakeword.model", openwakeword_model)

    colorama = types.ModuleType("colorama")
    colorama.init = lambda *args, **kwargs: None
    colorama.Fore = types.SimpleNamespace()
    colorama.Style = types.SimpleNamespace()
    _stub_module("colorama", colorama)

    scipy = types.ModuleType("scipy")
    scipy_signal = types.ModuleType("scipy.signal")
    scipy_signal.resample = lambda data, new_size: data
    scipy_signal.butter = lambda *args, **kwargs: (None, None)
    scipy_signal.filtfilt = lambda b, a, data: data
    scipy_signal.resample_poly = lambda data, up, down: data
    scipy.signal = scipy_signal
    _stub_module("scipy", scipy)
    _stub_module("scipy.signal", scipy_signal)

    pyaudio = types.ModuleType("pyaudio")
    pyaudio.paInt16 = 8
    pyaudio.PyAudio = type(
        "PyAudio",
        (),
        {
            "terminate": lambda self: None,
            "get_default_input_device_info": lambda self: {"index": 0},
            "open": lambda self, *args, **kwargs: types.SimpleNamespace(
                read=lambda *a, **k: b"",
                stop_stream=lambda: None,
                close=lambda: None,
                is_active=lambda: False,
            ),
        },
    )
    _stub_module("pyaudio", pyaudio)

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.hub = types.SimpleNamespace(load=lambda *args, **kwargs: (None, None))
    torch_multiprocessing = types.ModuleType("torch.multiprocessing")
    torch_multiprocessing.Value = lambda *args, **kwargs: types.SimpleNamespace(value=args[1] if len(args) > 1 else None)
    torch_multiprocessing.Event = type("Event", (), {})
    torch_multiprocessing.Queue = type("Queue", (), {})
    torch_multiprocessing.Process = type("Process", (), {})
    torch_multiprocessing.get_start_method = lambda allow_none=True: "spawn"
    torch_multiprocessing.set_start_method = lambda *args, **kwargs: None
    torch.multiprocessing = torch_multiprocessing
    _stub_module("torch", torch)
    _stub_module("torch.multiprocessing", torch_multiprocessing)

    soundfile = types.ModuleType("soundfile")
    soundfile.read = lambda *args, **kwargs: ([], 16000)
    _stub_module("soundfile", soundfile)

    halo = types.ModuleType("halo")
    halo.Halo = type("Halo", (), {"__init__": lambda self, *args, **kwargs: None, "start": lambda self: None, "stop": lambda self: None})
    _stub_module("halo", halo)

    pvporcupine = types.ModuleType("pvporcupine")
    pvporcupine.create = lambda *args, **kwargs: types.SimpleNamespace(process=lambda pcm: -1, delete=lambda: None)
    _stub_module("pvporcupine", pvporcupine)

    webrtcvad = types.ModuleType("webrtcvad")
    webrtcvad.Vad = type("Vad", (), {"__init__": lambda self, *args, **kwargs: None, "is_speech": lambda self, *args, **kwargs: False})
    _stub_module("webrtcvad", webrtcvad)

    websocket = types.ModuleType("websocket")
    websocket.WebSocketApp = type("WebSocketApp", (), {})
    websocket.ABNF = types.SimpleNamespace(OPCODE_BINARY=2)
    _stub_module("websocket", websocket)


_install_client_dependency_stubs()
audio_recorder_client = importlib.import_module("RealtimeSTT.audio_recorder_client")
model_resolver = importlib.import_module("RealtimeSTT.asr.model_resolver")


class WhisperCppSupportTests(unittest.TestCase):
    def test_client_start_server_includes_backend_arguments(self):
        client = audio_recorder_client.AudioToTextRecorderClient.__new__(audio_recorder_client.AudioToTextRecorderClient)
        client.model = "base.en"
        client.backend = "whisper.cpp"
        client.realtime_model_type = "tiny.en"
        client.whisper_cpp_model_path = "/models/main.bin"
        client.whisper_cpp_realtime_model_path = "/models/rt.bin"
        client.whisper_cpp_threads = 8
        client.whisper_cpp_realtime_threads = 4
        client.whisper_cpp_acceleration = "metal"
        client.whisper_cpp_coreml_encoder_path = None
        client.whisper_cpp_openvino_encoder_path = None
        client.whisper_cpp_openvino_device = "CPU"
        client.whisper_cpp_openvino_cache_dir = None
        client.whisper_cpp_no_context_realtime = False
        client.whisper_cpp_single_segment_realtime = True
        client.download_root = None
        client.batch_size = 16
        client.realtime_batch_size = 16
        client.init_realtime_after_seconds = 0.2
        client.initial_prompt_realtime = ""
        client.debug_mode = False
        client.language = "en"
        client.silero_sensitivity = 0.4
        client.silero_use_onnx = False
        client.webrtc_sensitivity = 3
        client.min_length_of_recording = 0.5
        client.min_gap_between_recordings = 0
        client.realtime_processing_pause = 0.2
        client.early_transcription_on_silence = 0
        client.silero_deactivity_detection = False
        client.beam_size = 5
        client.beam_size_realtime = 3
        client.wake_words = ""
        client.wake_words_sensitivity = 0.6
        client.wake_word_timeout = 5.0
        client.wake_word_activation_delay = 0.0
        client.wakeword_backend = "pvporcupine"
        client.openwakeword_model_paths = None
        client.openwakeword_inference_framework = "onnx"
        client.wake_word_buffer_duration = 0.1
        client.use_main_model_for_realtime = False
        client.use_extended_logging = False
        client.control_url = "ws://127.0.0.1:8011"
        client.data_url = "ws://127.0.0.1:8012"
        client.initial_prompt = ""

        with patch("RealtimeSTT.audio_recorder_client.subprocess.Popen") as popen:
            client.start_server()

        args = popen.call_args.args[0]
        self.assertIn("--backend", args)
        self.assertIn("whisper.cpp", args)
        self.assertIn("--whisper_cpp_model_path", args)
        self.assertIn("/models/main.bin", args)
        self.assertIn("--whisper_cpp_acceleration", args)
        self.assertIn("metal", args)
        self.assertIn("--whisper_cpp_no_context_realtime", args)
        self.assertIn("false", args)

    def test_whisper_cpp_known_model_downloads_and_verifies_checksum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = "base.en"
            model_path = Path(tmpdir) / "ggml-base.en.bin"
            payload = b"hello-whisper"

            with patch.object(model_resolver, "WHISPER_CPP_KNOWN_HASHES", {model_name: "d9e485310d5fb27c134aa9654d3e2e8ddcf252b5"}):
                with patch("RealtimeSTT.asr.model_resolver.urllib.request.urlretrieve") as urlretrieve:
                    urlretrieve.side_effect = lambda url, target: Path(target).write_bytes(payload)
                    resolved = model_resolver.resolve_model_identifier(model_name, tmpdir, backend="whisper.cpp")

            self.assertEqual(resolved, str(model_path))
            self.assertTrue(model_path.exists())


if __name__ == "__main__":
    unittest.main()
