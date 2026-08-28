import copy
import json
import tempfile
import unittest
from pathlib import Path

from runtime_helpers import (
    DEFAULT_OLLAMA_PROMPT_TOKENS,
    bmo_runtime_defaults,
    calculate_cefr_level,
    calculate_overall_score,
    compact_context,
    contains_wake_word,
    detect_mode_selection,
    emergency_context,
    estimate_prompt_tokens,
    extract_action,
    interpolate_wake_word_name,
    load_voice_sample_rate,
    missing_wake_word_warning,
    normalize_action,
    normalize_history,
    split_tts_segments,
    validate_voice_model,
    load_history_file,
    load_skill,
    ollama_prompt_budget,
    parse_test_response,
    save_history_file,
    validate_test_result,
    validate_test_scores,
    wake_word_input_frame_size,
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


class RuntimeHelperAudioTests(unittest.TestCase):
    def test_wake_word_frame_size_preserves_80ms_at_common_rates(self):
        self.assertEqual(wake_word_input_frame_size(16000), 1280)
        self.assertEqual(wake_word_input_frame_size(48000), 3840)
        self.assertEqual(wake_word_input_frame_size(44100), 3528)

    def test_wake_word_frame_size_rejects_invalid_rates(self):
        self.assertEqual(wake_word_input_frame_size(0), 1280)
        self.assertEqual(wake_word_input_frame_size("bad"), 1280)

    def test_wake_word_frame_size_supports_custom_target_frame(self):
        self.assertEqual(wake_word_input_frame_size(48000, 16000, 640), 1920)


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
                "text_model": "qwen3.5:4b",
                "vision_model": "qwen3.5:4b",
                "voice_model": "piper/en_GB-semaine-medium.onnx",
                "chat_memory": True,
                "camera_rotation": 180,
                "system_prompt_extras": "",
                "input_device": None,
                "input_sample_rate": 44100,
                "silence_threshold": 0.006,
                "recording_settle_delay": 0.15,
                "silence_duration": 0.8,
                "wake_word_name": "Hey BMO",
                "ollama_context_tokens": 2048,
                "ollama_output_tokens": 40,
                "ollama_prompt_tokens": 1400,
            },
        )

    def test_missing_wake_word_warning_keeps_ptt_available(self):
        self.assertEqual(
            missing_wake_word_warning("./wakeword.onnx", "Hey BMO"),
            "[WARNING] Wake-word model missing: ./wakeword.onnx. "
            "Add user-supplied wakeword.onnx trained for 'Hey BMO'; "
            "push-to-talk remains available.",
        )

    def test_wake_word_template_keeps_action_json_machine_parseable(self):
        template = (
            "Say {wake_word_name}. "
            'JSON: {"action": "ACTION_NAME", "value": "short request value"}'
        )

        self.assertEqual(
            interpolate_wake_word_name(template, "Hey BMO"),
            (
                "Say Hey BMO. "
                'JSON: {"action": "ACTION_NAME", "value": "short request value"}'
            ),
        )


class RuntimeHelperQwenModeTests(unittest.TestCase):
    def test_mode_selection_accepts_standalone_mode_word(self):
        self.assertEqual(detect_mode_selection("Can we use practice mode?"), "practice")
        self.assertEqual(detect_mode_selection("TEST MODE please"), "test")
        self.assertEqual(detect_mode_selection("Let us practice."), "practice")
        self.assertEqual(detect_mode_selection("I want to test my English"), "test")

    def test_mode_selection_rejects_ambiguous_or_embedded_words(self):
        self.assertIsNone(detect_mode_selection("practice mode or test mode"))
        self.assertIsNone(detect_mode_selection("contest"))

    def test_wake_word_matching_is_case_insensitive_and_phrase_bounded(self):
        self.assertTrue(contains_wake_word("hey bmo, practice mode", "Hey BMO"))
        self.assertTrue(contains_wake_word("Please HEY BMO!", "Hey BMO"))
        self.assertFalse(contains_wake_word("say hey bmore", "Hey BMO"))
        self.assertFalse(contains_wake_word("hey bmo", ""))

    def test_skill_loading_uses_bundled_mode_files(self):
        self.assertIn("A1–A2", load_skill("practice"))
        self.assertIn("exactly one JSON object", load_skill("test"))
        self.assertEqual(load_skill("unknown"), "")

    def test_context_budget_reserves_output_tokens(self):
        self.assertEqual(ollama_prompt_budget(4096, 384), 2784)
        self.assertEqual(ollama_prompt_budget(1000, 900), 75)
        self.assertEqual(
            ollama_prompt_budget(
                2048, 40, prompt_tokens=DEFAULT_OLLAMA_PROMPT_TOKENS
            ),
            1400,
        )
        self.assertEqual(ollama_prompt_budget(2048, 40, prompt_tokens=9999), 1506)

    def test_token_estimate_is_conservative_for_mixed_language(self):
        self.assertGreater(
            estimate_prompt_tokens("Hello learner. สวัสดี"),
            estimate_prompt_tokens("Hello learner."),
        )

    def test_compaction_preserves_required_messages_and_drops_oldest_turns(self):
        turns = []
        for number in range(1, 10):
            turns.extend([
                {"role": "user", "content": f"old question {number} " + "x" * 100},
                {"role": "assistant", "content": f"old answer {number} " + "y" * 100},
            ])
        result = compact_context(
            "core rules",
            "practice skill",
            "mode state",
            {"role": "user", "content": "current question"},
            turns,
            context_tokens=180,
            output_tokens=20,
        )
        contents = [message["content"] for message in result]
        self.assertEqual(contents[:3], ["core rules", "practice skill", "mode state"])
        self.assertEqual(contents[-1], "current question")
        self.assertNotIn("old question 1 " + "x" * 100, contents)
        self.assertTrue(any("old question 9" in content for content in contents))

    def test_compaction_honors_explicit_prompt_budget(self):
        result = compact_context(
            "core", "skill", "state", {"role": "user", "content": "now"},
            [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ],
            context_tokens=4096,
            output_tokens=48,
            prompt_tokens=20,
        )
        self.assertEqual(
            [message["content"] for message in result],
            ["core", "skill", "state", "now"],
        )

    def test_default_budget_allows_recent_mode_history_after_required_context(self):
        # Required mode prompts are larger than the old 320-token cap. The
        # configured 1,400-token cap must still retain a complete recent pair.
        required_core = "core safety rules " + ("x" * 600)
        required_skill = "practice skill " + ("y" * 300)
        mode_state = "mode state " + ("z" * 150)
        recent_turn = [
            {"role": "user", "content": "My name is Sam."},
            {"role": "assistant", "content": "Nice to meet you, Sam!"},
        ]

        result = compact_context(
            required_core,
            required_skill,
            mode_state,
            {"role": "user", "content": "I like apples."},
            recent_turn,
            context_tokens=2048,
            output_tokens=40,
            prompt_tokens=DEFAULT_OLLAMA_PROMPT_TOKENS,
        )

        contents = [message["content"] for message in result]
        self.assertIn("My name is Sam.", contents)
        self.assertIn("Nice to meet you, Sam!", contents)
        self.assertLessEqual(estimate_prompt_tokens(result), 1400)

    def test_practice_context_keeps_recent_complete_turn_only(self):
        result = compact_context(
            "core", "practice", "practice state", {"role": "user", "content": "now"},
            [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "latest"},
                {"role": "assistant", "content": "latest answer"},
            ],
            context_tokens=80,
            output_tokens=10,
        )
        self.assertEqual(result[-1]["content"], "now")
        self.assertIn("latest", [message["content"] for message in result])

    def test_compaction_drops_newest_pair_when_it_exceeds_budget(self):
        result = compact_context(
            "core", "skill", "state", {"role": "user", "content": "now"},
            [
                {"role": "user", "content": "short"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "long " + "x" * 300},
                {"role": "assistant", "content": "long answer " + "y" * 300},
            ],
            context_tokens=100,
            output_tokens=10,
        )
        contents = [message["content"] for message in result]
        self.assertNotIn("long " + "x" * 300, contents)
        self.assertNotIn("long answer " + "y" * 300, contents)

    def test_test_context_contains_runtime_state_not_persisted_memory(self):
        result = compact_context(
            "core", "test skill", '{"test_question_index":2}',
            {"role": "user", "content": "answer"},
            [], context_tokens=100, output_tokens=10,
        )
        contents = [message["content"] for message in result]
        self.assertIn('{"test_question_index":2}', contents)
        self.assertNotIn("old persisted memory", contents)

    def test_emergency_context_has_no_conversation_turns(self):
        result = emergency_context(
            "core", "test skill", "test state", {"role": "user", "content": "current"}
        )
        self.assertEqual(
            [message["content"] for message in result],
            ["core", "test skill", "test state", "current"],
        )

    def test_test_scores_clamp_and_cefr_mapping(self):
        self.assertEqual(
            validate_test_scores({"grammar": -2, "vocabulary": 9, "comprehension": 3.8}),
            {"grammar": 0, "vocabulary": 5, "comprehension": 3},
        )
        self.assertEqual(
            validate_test_result({
                "speech": "Good.", "grammar": 6, "vocabulary": 2,
                "comprehension": -1, "done": False, "extra": "drop",
            }),
            {"speech": "Good.", "grammar": 5, "vocabulary": 2, "comprehension": 0, "done": False},
        )
        self.assertEqual(calculate_overall_score({
            "grammar": [2, 4], "vocabulary": [3, 5], "comprehension": [4, 4],
        }), 3.67)
        self.assertEqual(calculate_cefr_level(2.99), "A1")
        self.assertEqual(calculate_cefr_level(3.0), "A2")
        self.assertIsNone(parse_test_response('{"speech":"broken"'))


if __name__ == "__main__":
    unittest.main()
