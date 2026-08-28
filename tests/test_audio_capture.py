import threading
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import agent


class _OverflowStatus:
    input_overflow = True

    def __str__(self):
        return "input overflow"


class _CallbackStream:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def __enter__(self):
        callback = self.kwargs["callback"]
        # Variable callback blocks model blocksize=0. Four blocks contain one
        # complete 80 ms frame at 48 kHz plus a short remainder.
        for _ in range(4):
            data = np.full((1024, 1), 1000, dtype=np.int16)
            callback(data, len(data), None, _OverflowStatus())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _WakeModel:
    prediction_buffer = {"wake": [0.9]}

    def __init__(self):
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame.copy())

    def reset(self):
        pass


class AudioCaptureTests(unittest.TestCase):
    def test_callback_assembles_full_frame_and_recovers_overflow(self):
        bot = agent.BotGUI.__new__(agent.BotGUI)
        bot.ptt_event = threading.Event()
        bot.oww_model = _WakeModel()
        _CallbackStream.instances = []

        with patch.object(agent.sd, "InputStream", _CallbackStream):
            with patch.object(agent.select, "select", return_value=([], [], [])):
                with patch("builtins.print") as print_mock:
                    bot._listen_loop(
                        {
                            "samplerate": 48000,
                            "channels": 1,
                            "dtype": "int16",
                            "blocksize": 0,
                            "latency": "high",
                            "device": None,
                        },
                        input_chunk_size=3840,
                        target_chunk_size=1280,
                        use_resampling=True,
                    )

        self.assertEqual(len(bot.oww_model.frames), 1)
        self.assertEqual(bot.oww_model.frames[0].shape, (1280,))
        self.assertEqual(bot.oww_model.frames[0].dtype, np.int16)
        self.assertEqual(_CallbackStream.instances[0].kwargs["blocksize"], 0)
        messages = [call.args[0] for call in print_mock.call_args_list]
        self.assertTrue(any("Input timing warning recovered" in msg for msg in messages))

    def test_configured_duration_uses_safe_bounds(self):
        with patch.dict(
            agent.CURRENT_CONFIG,
            {
                "recording_settle_delay": "not-a-number",
                "silence_duration": 45,
            },
            clear=False,
        ):
            self.assertEqual(
                agent.configured_duration("recording_settle_delay", 0.15, 2.0),
                0.15,
            )
            self.assertEqual(
                agent.configured_duration("silence_duration", 0.8),
                30.0,
            )

    def test_save_audio_buffer_dispatches_ack_without_waiting(self):
        bot = agent.BotGUI.__new__(agent.BotGUI)
        bot.get_random_sound = lambda directory: "ack.wav"
        bot.play_sound_async = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/input.wav"
            result = bot.save_audio_buffer(
                [np.array([[0.1], [-0.1]], dtype=np.float32)],
                filename,
                samplerate=16000,
            )

            self.assertEqual(result, filename)
            bot.play_sound_async.assert_called_once_with("ack.wav")


if __name__ == "__main__":
    unittest.main()
