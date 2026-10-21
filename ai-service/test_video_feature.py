import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main
from fastapi import HTTPException


class VideoFeatureTests(unittest.TestCase):
    def test_disabled_video_does_not_degrade_health(self):
        with patch.object(main, "VIDEO_ENABLED", False), patch.object(main, "is_llm_configured", return_value=True), patch.object(main, "is_llm_connected", return_value=True), patch.object(main, "_tts_health", return_value={"ready": False}):
            health = asyncio.run(main.health_check())
        self.assertEqual(health["status"], "ok")
        self.assertFalse(health["mp4_ready"])
        self.assertFalse(health["video_enabled"])

    def test_enabled_video_requires_dependencies(self):
        with patch.object(main, "VIDEO_ENABLED", True), patch.object(main, "is_llm_configured", return_value=True), patch.object(main, "is_llm_connected", return_value=True), patch.object(main, "_tts_health", return_value={"ready": False}):
            health = asyncio.run(main.health_check())
        self.assertEqual(health["status"], "degraded")

    def test_disabled_video_rejects_generation_before_rendering(self):
        with patch.object(main, "VIDEO_ENABLED", False), patch.object(main, "_current_user_id", return_value="user"), patch.object(main, "generate_video_from_script") as generate:
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(main.create_video(None, "analysis", None))
        self.assertEqual(caught.exception.status_code, 503)
        generate.assert_not_called()

    def test_public_config_exposes_switch(self):
        with patch.object(main, "VIDEO_ENABLED", False), patch.object(main, "supabase_public_config", return_value={"enabled": False}):
            config = asyncio.run(main.auth_config())
        self.assertFalse(config["video_enabled"])
