from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qwen_video_study.py"
SPEC = importlib.util.spec_from_file_location("qwen_video_study", SCRIPT)
qwen_video_study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = qwen_video_study
SPEC.loader.exec_module(qwen_video_study)


class QwenVideoStudyTests(unittest.TestCase):
    def test_extract_json_block_accepts_fenced_json(self):
        data = qwen_video_study.extract_json_block('```json\n{"a": 1, "b": [2]}\n```')
        self.assertEqual(data["a"], 1)
        self.assertEqual(data["b"], [2])

    def test_normalize_analysis_rewrites_unknown_frame_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_path = root / "frame-001.jpg"
            frame_path.write_bytes(b"jpg")
            frames = [
                qwen_video_study.Frame(
                    frame_id="frame-001",
                    timestamp=1.0,
                    time_label="00:01",
                    path=frame_path,
                    segment_index=0,
                )
            ]
            analysis = {
                "knowledge_points": [{"title": "x", "frame_id": "frame-999"}],
                "timeline": [{"title": "y", "frame_id": ""}],
            }
            normalized = qwen_video_study.normalize_analysis(analysis, frames)
            self.assertEqual(normalized["knowledge_points"][0]["frame_id"], "frame-001")
            self.assertEqual(normalized["timeline"][0]["frame_id"], "frame-001")
            self.assertTrue(normalized["normalization_notes"])

    def test_segment_message_contains_image_payload_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_path = Path(tmp) / "frame-001.jpg"
            frame_path.write_bytes(b"\xff\xd8fakejpg")
            frame = qwen_video_study.Frame(
                frame_id="frame-001",
                timestamp=12,
                time_label="00:12",
                path=frame_path,
                segment_index=0,
            )
            messages = qwen_video_study.build_segment_messages(
                video_title="Demo",
                segment_index=0,
                start=0,
                end=30,
                frames=[frame],
                transcript_text="hello",
            )
            dumped = json.dumps(messages, ensure_ascii=False)
            self.assertIn("data:image/jpeg;base64", dumped)
            self.assertIn("frame-001", dumped)
            self.assertNotIn("sk-", dumped)

    def test_validate_html_media_finds_missing_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "assets").mkdir()
            (root / "assets" / "ok.jpg").write_bytes(b"jpg")
            html = root / "index.html"
            html.write_text('<img src="assets/ok.jpg"><video poster="assets/missing.jpg"></video>', encoding="utf-8")
            missing = qwen_video_study.validate_html_media(html, root)
            self.assertEqual(missing, ["assets/missing.jpg"])


if __name__ == "__main__":
    unittest.main()
