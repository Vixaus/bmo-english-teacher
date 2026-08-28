import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import agent


class CameraHistoryTests(unittest.TestCase):
    def make_bot(self, active_mode="practice"):
        bot = agent.BotGUI.__new__(agent.BotGUI)
        bot.permanent_memory = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
        bot.session_memory = []
        bot.active_mode = active_mode
        bot.active_mode_context = []
        bot.test_question_index = 0
        bot.test_scores = {
            "grammar": [],
            "vocabulary": [],
            "comprehension": [],
        }
        bot.interrupted = threading.Event()
        bot.thinking_sound_active = threading.Event()
        bot.tts_queue_lock = threading.Lock()
        bot.tts_queue = []
        bot.set_state = Mock()
        bot.append_to_text = Mock()
        bot.wait_for_tts = Mock()
        bot._run_thinking_sound_loop = Mock()
        bot.execute_action_and_get_result = Mock(
            return_value="IMAGE_CAPTURE_TRIGGERED"
        )
        bot.capture_image = Mock(return_value="camera.jpg")
        bot.commit_memory = agent.BotGUI.commit_memory.__get__(bot)
        bot._append_mode_turn = agent.BotGUI._append_mode_turn.__get__(bot)
        return bot

    def test_initial_mode_selection_requires_wake_word(self):
        bot = self.make_bot(active_mode=None)
        bot._request_qwen_text = Mock(return_value=("Please say Hey BMO to begin.", False))
        bot._deliver_response = Mock()

        bot.chat_and_respond("Practice Mode")

        self.assertIsNone(bot.active_mode)
        bot._deliver_response.assert_called_once_with(
            "Please say Hey BMO to begin.",
            user_text="Practice Mode",
            persist=True,
            img_path=None,
        )

    def test_mode_selection_after_greeting_without_repeating_wake_word(self):
        bot = self.make_bot(active_mode=None)
        bot.greeting_acknowledged = False
        bot._request_qwen_text = Mock()
        bot._deliver_response = Mock()

        bot.chat_and_respond("Hey BMO")
        self.assertTrue(bot.greeting_acknowledged)
        self.assertIsNone(bot.active_mode)

        bot.chat_and_respond("Practice Mode")

        self.assertEqual(bot.active_mode, "practice")
        self.assertEqual(bot._request_qwen_text.call_count, 0)
        self.assertEqual(
            [call.args[0] for call in bot._deliver_response.call_args_list],
            [
                "Hey BMO! Please choose a mode:\n"
                "  Practice Mode\n"
                "  Test Mode",
                "Great! Let us practice. What is your name?",
            ],
        )

    def test_physical_wake_accepts_greeting_and_opens_mode_menu(self):
        bot = self.make_bot(active_mode=None)
        bot.greeting_acknowledged = False
        bot._deliver_response = Mock()

        bot._handle_wake_trigger()

        self.assertTrue(bot.greeting_acknowledged)
        bot._deliver_response.assert_called_once_with(
            "Hey BMO! Please choose a mode:\n"
            "  Practice Mode\n"
            "  Test Mode",
            user_text=None,
            persist=False,
            img_path=None,
        )

    def test_physical_wake_during_active_mode_only_accepts_greeting(self):
        bot = self.make_bot(active_mode="practice")
        bot.greeting_acknowledged = False
        bot._deliver_response = Mock()

        bot._handle_wake_trigger()

        self.assertTrue(bot.greeting_acknowledged)
        bot._deliver_response.assert_not_called()

    def test_interactive_ollama_chat_disables_thinking(self):
        with patch.object(agent.ollama, "chat", return_value=[]) as chat:
            agent.ollama_chat_stream([], model="qwen3.5:4b")

        self.assertFalse(chat.call_args.kwargs["think"])
        self.assertEqual(chat.call_args.kwargs["keep_alive"], -1)
        self.assertNotIn("keep_alive", chat.call_args.kwargs["options"])

    def test_ollama_timing_logs_only_final_stats(self):
        chunks = [
            {"message": {"content": "Hi"}},
            {
                "done": True,
                "total_duration": 2_000_000,
                "prompt_eval_count": 10,
                "prompt_eval_duration": 1_000_000,
                "eval_count": 2,
                "eval_duration": 1_000_000,
            },
        ]
        with patch("builtins.print") as print_mock:
            self.assertEqual(
                list(agent._instrument_ollama_stream(chunks, "qwen3.5:4b")),
                chunks,
            )

        timing = [call.args[0] for call in print_mock.call_args_list]
        self.assertEqual(len(timing), 1)
        self.assertIn("prompt_tokens=10", timing[0])
        self.assertIn("output_tokens=2", timing[0])

    def test_interruption_during_malformed_test_retry_aborts_delivery(self):
        bot = self.make_bot(active_mode="test")
        bot._request_qwen_text = Mock(return_value=("malformed", False))

        def interrupt_during_retry(_user_message):
            bot.interrupted.set()
            return json.dumps({
                "speech": "Good.",
                "grammar": 4,
                "vocabulary": 4,
                "comprehension": 4,
                "done": False,
            })

        bot._retry_malformed_test_response = Mock(side_effect=interrupt_during_retry)
        bot._deliver_response = Mock()

        bot.chat_and_respond("I like apples.")

        bot._deliver_response.assert_not_called()
        self.assertEqual(bot.session_memory, [])
        self.assertEqual(bot.tts_queue, [])

    def test_camera_turn_persists_original_question_and_description(self):
        bot = self.make_bot()
        question = "What is in this picture?"
        description = "I see a red ball."
        action_response = '{"action":"capture_image","value":"picture"}'
        original_deliver = agent.BotGUI._deliver_response.__get__(bot)
        bot._deliver_response = Mock(wraps=original_deliver)
        bot._request_qwen_text = Mock(return_value=(action_response, False))

        with TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory.json"
            with patch.object(agent, "MEMORY_FILE", str(memory_path)):
                with patch.object(
                    agent,
                    "ollama_chat_stream",
                    return_value=[{"message": {"content": description}}],
                ) as chat_stream:
                    bot.chat_and_respond(question)

            bot._deliver_response.assert_called_once_with(
                description,
                user_text=question,
                persist=True,
                img_path="camera.jpg",
            )
            self.assertEqual(
                bot.active_mode_context,
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": description},
                ],
            )
            history = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(history[1:], bot.active_mode_context)
            self.assertNotIn(action_response, json.dumps(history))
            self.assertNotIn("Describe this image", json.dumps(history))
            chat_stream.assert_called_once()


if __name__ == "__main__":
    unittest.main()
