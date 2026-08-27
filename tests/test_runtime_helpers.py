import copy
import json
import tempfile
import unittest
from pathlib import Path

from runtime_helpers import (
    bmo_runtime_defaults,
    extract_action,
    interpolate_wake_word_name,
    load_voice_sample_rate,
    missing_wake_word_warning,
    normalize_action,
    normalize_history,
    split_tts_segments,
    validate_voice_model,
    load_history_file,
    save_history_file,
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

    def test_extract_action_rejects_unsupported_action_object(self):
        self.assertIsNone(
            extract_action('{"action": "launch_rocket", "value": "now"}')
        )

    def test_normalize_action_rejects_missing_value(self):
        self.assertIsNone(normalize_action({"action": "search_web"}))

    def test_normalize_action_rejects_query_without_value(self):
        self.assertIsNone(normalize_action({"action": "search_web", "query": "cats"}))

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
    def test_validate_voice_model_rejects_tiny_not_found_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "bmo-custom.onnx"
            model_path.write_bytes(b"Not Found")
            self.assertFalse(validate_voice_model(str(model_path))[0])

    def test_validate_voice_model_requires_valid_adjacent_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "bmo-custom.onnx"
            model_path.write_bytes(b"0" * 2048)
            Path(f"{model_path}.json").write_text(
                json.dumps({"audio": {"sample_rate": 22050}}), encoding="utf-8"
            )
            self.assertEqual(validate_voice_model(str(model_path)), (True, ""))

    def test_validate_voice_model_rejects_malformed_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "bmo-custom.onnx"
            model_path.write_bytes(b"0" * 2048)
            Path(f"{model_path}.json").write_text("broken", encoding="utf-8")
            self.assertFalse(validate_voice_model(str(model_path))[0])
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
    def test_disabled_history_does_not_read_or_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            self.assertEqual(load_history_file(str(path), "fresh", enabled=False), [
                {"role": "system", "content": "fresh"}
            ])
            self.assertFalse(save_history_file(str(path), [{"role": "system", "content": "fresh"}], enabled=False))
            self.assertFalse(path.exists())

    def test_history_file_loads_malformed_data_safely_and_saves_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(load_history_file(str(path), "fresh"), [
                {"role": "system", "content": "fresh"}
            ])
            self.assertTrue(save_history_file(str(path), [
                {"role": "system", "content": "fresh"},
                {"role": "user", "content": "hello"},
            ]))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))[1]["content"], "hello")
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


class RuntimeHelperBmoIdentityTests(unittest.TestCase):
    def test_bmo_runtime_defaults_match_canonical_config(self):
        self.assertEqual(
            bmo_runtime_defaults(),
            {
                "text_model": "qwen2.5:3b",
                "vision_model": "moondream",
                "voice_model": "voices/bmo-custom.onnx",
                "chat_memory": True,
                "camera_rotation": 180,
                "system_prompt_extras": "",
                "input_device": None,
                "input_sample_rate": 44100,
                "wake_word_name": "Hello BMO",
            },
        )

    def test_missing_wake_word_warning_keeps_ptt_available(self):
        self.assertEqual(
            missing_wake_word_warning("./wakeword.onnx", "Hello BMO"),
            "[WARNING] Wake-word model missing: ./wakeword.onnx. "
            "Add user-supplied wakeword.onnx trained for 'Hello BMO'; "
            "push-to-talk remains available.",
        )

    def test_wake_word_template_keeps_action_json_machine_parseable(self):
        template = (
            "Say {wake_word_name}. "
            'JSON: {"action": "ACTION_NAME", "value": "short request value"}'
        )

        self.assertEqual(
            interpolate_wake_word_name(template, "Hello BMO"),
            (
                "Say Hello BMO. "
                'JSON: {"action": "ACTION_NAME", "value": "short request value"}'
            ),
        )


if __name__ == "__main__":
    unittest.main()
