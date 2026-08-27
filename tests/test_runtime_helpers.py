import copy
import json
import tempfile
import unittest
from pathlib import Path

from runtime_helpers import (
    extract_action,
    load_voice_sample_rate,
    normalize_action,
    normalize_history,
    split_tts_segments,
)


class RuntimeHelperActionTests(unittest.TestCase):
    def test_extract_action_parses_complete_json_and_normalizes_alias(self):
        result = extract_action('  {"action": "google", "value": "robot news"}  ')

        self.assertEqual(result, {"action": "search_web", "value": "robot news"})

    def test_extract_action_rejects_malformed_json(self):
        self.assertIsNone(
            extract_action('{"action": "search_web", "value": "robot news"')
        )

    def test_extract_action_rejects_plain_text_with_action_marker(self):
        self.assertIsNone(
            extract_action(
                'Please action: {"action": "search_web", "value": "robot news"}'
            )
        )

    def test_normalize_action_rejects_missing_value(self):
        self.assertIsNone(normalize_action({"action": "search_web"}))

    def test_normalize_action_maps_aliases(self):
        result = normalize_action({"action": "check_time", "value": "now"})

        self.assertEqual(result, {"action": "get_time", "value": "now"})


class RuntimeHelperTtsTests(unittest.TestCase):
    def test_split_tts_segments_handles_english_only(self):
        self.assertEqual(
            split_tts_segments("Hello there"),
            [("english", "Hello there")],
        )

    def test_split_tts_segments_handles_thai_only(self):
        self.assertEqual(
            split_tts_segments("สวัสดีครับ"),
            [("thai", "สวัสดีครับ")],
        )

    def test_split_tts_segments_preserves_mixed_order(self):
        self.assertEqual(
            split_tts_segments("Hello สวัสดี world"),
            [("english", "Hello"), ("thai", "สวัสดี"), ("english", "world")],
        )


class RuntimeHelperVoiceTests(unittest.TestCase):
    def test_load_voice_sample_rate_reads_nested_audio_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "voice.onnx"
            metadata_path = Path(f"{model_path}.json")
            model_path.write_text("dummy", encoding="utf-8")
            metadata_path.write_text(
                json.dumps({"audio": {"sample_rate": 22050}}),
                encoding="utf-8",
            )

            self.assertEqual(load_voice_sample_rate(str(model_path)), 22050)

    def test_load_voice_sample_rate_reads_top_level_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "voice.onnx"
            metadata_path = Path(f"{model_path}.json")
            model_path.write_text("dummy", encoding="utf-8")
            metadata_path.write_text(json.dumps({"sample_rate": 44100}), encoding="utf-8")

            self.assertEqual(load_voice_sample_rate(str(model_path)), 44100)

    def test_load_voice_sample_rate_returns_none_when_metadata_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "voice.onnx"
            model_path.write_text("dummy", encoding="utf-8")

            self.assertIsNone(load_voice_sample_rate(str(model_path)))

    def test_load_voice_sample_rate_returns_none_for_malformed_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "voice.onnx"
            metadata_path = Path(f"{model_path}.json")
            model_path.write_text("dummy", encoding="utf-8")
            metadata_path.write_text("{not json", encoding="utf-8")

            self.assertIsNone(load_voice_sample_rate(str(model_path)))

    def test_load_voice_sample_rate_returns_none_for_invalid_sample_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "voice.onnx"
            metadata_path = Path(f"{model_path}.json")
            model_path.write_text("dummy", encoding="utf-8")
            metadata_path.write_text(
                json.dumps({"audio": {"sample_rate": 0}}),
                encoding="utf-8",
            )

            self.assertIsNone(load_voice_sample_rate(str(model_path)))


class RuntimeHelperHistoryTests(unittest.TestCase):
    def test_normalize_history_replaces_system_message_and_limits_recent_messages(self):
        raw_history = [
            {"role": "system", "content": "old prompt"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "skip me"},
            {"role": "user", "content": "u2"},
            {"role": "system", "content": "stale prompt"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
            {"role": "assistant", "content": "a4"},
            {"role": "user", "content": "u5"},
            {"role": "assistant", "content": "a5"},
            {"role": "assistant", "content": None},
            "not a message",
        ]
        original = copy.deepcopy(raw_history)
        system_prompt = "fresh system prompt"

        result = normalize_history(raw_history, system_prompt, limit=4)

        self.assertEqual(
            result,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "u4"},
                {"role": "assistant", "content": "a4"},
                {"role": "user", "content": "u5"},
                {"role": "assistant", "content": "a5"},
            ],
        )
        self.assertEqual(raw_history, original)


if __name__ == "__main__":
    unittest.main()
