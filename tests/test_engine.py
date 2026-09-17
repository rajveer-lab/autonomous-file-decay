"""End-to-end tests: real files on disk, real sqlite store, tight thresholds
so a handful of backdated files exercise the whole pipeline in one pass."""
import os
import tempfile
import time
import unittest
from pathlib import Path

from filedecay import actions, config, engine
from filedecay.store import Store

DAY = 86400


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        config.init_config(self.root)
        # tight thresholds so tests run in milliseconds instead of days
        cfg_path = config.config_path(self.root)
        cfg_path.write_text(
            "[thresholds]\n"
            "compress_after_days = 1\n"
            "archive_after_days = 3\n"
            "quarantine_after_days = 5\n"
            "delete_after_days = 1\n"
            "[scan]\n"
            'ignore = [".git", ".decay"]\n'
            "pinned = []\n"
        )
        self.cfg = config.load_config(self.root)
        self.store = Store(self.cfg.db_path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _write(self, rel: str, content: str, age_days: float = 0):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if age_days:
            ts = time.time() - age_days * DAY
            os.utime(path, (ts, ts))
        return path

    def test_fresh_file_stays_active(self):
        self._write("fresh.txt", "hi", age_days=0)
        engine.run_pass(self.cfg, self.store, apply=True)
        row = self.store.get_file("fresh.txt")
        self.assertEqual(row["state"], "active")

    def test_full_pipeline_in_one_pass(self):
        self._write("keep_active.txt", "a", age_days=0)
        self._write("gets_compressed.txt", "b", age_days=2)
        self._write("gets_archived.txt", "c", age_days=4)
        self._write("gets_quarantined.txt", "d", age_days=6)

        # first pass: newly discovered files just get tracked as active
        engine.run_pass(self.cfg, self.store, apply=True)
        # second pass: now their idle time is evaluated against thresholds
        engine.run_pass(self.cfg, self.store, apply=True)

        self.assertEqual(self.store.get_file("keep_active.txt")["state"], "active")
        self.assertEqual(self.store.get_file("gets_compressed.txt")["state"], "compressed")
        self.assertEqual(self.store.get_file("gets_archived.txt")["state"], "archived")
        self.assertEqual(self.store.get_file("gets_quarantined.txt")["state"], "quarantined")

        # archived file should actually be sitting under .decay/archive/
        archived = self.store.get_file("gets_archived.txt")
        self.assertTrue((self.root / archived["current_rel_path"]).exists())
        self.assertIn("archive", archived["current_rel_path"])

    def test_dry_run_changes_nothing(self):
        self._write("old.txt", "x", age_days=10)
        planned = engine.run_pass(self.cfg, self.store, apply=False)
        self.assertTrue(any(p.rel_path == "old.txt" for p in planned))
        # dry run must not create a store row or touch the file on disk
        self.assertTrue((self.root / "old.txt").exists())
        self.assertIsNone(self.store.get_file("old.txt"))

    def test_quarantine_then_delete_after_grace_period(self):
        self._write("doomed.txt", "gone soon", age_days=10)
        engine.run_pass(self.cfg, self.store, apply=True)
        engine.run_pass(self.cfg, self.store, apply=True)
        row = self.store.get_file("doomed.txt")
        self.assertEqual(row["state"], "quarantined")
        quarantined_path = self.root / row["current_rel_path"]
        self.assertTrue(quarantined_path.exists())

        # simulate the grace period elapsing by backdating state_changed_at
        self.store.conn.execute(
            "UPDATE files SET state_changed_at = ? WHERE id = ?",
            (time.time() - 2 * DAY, row["id"]),
        )
        self.store.conn.commit()

        engine.run_pass(self.cfg, self.store, apply=True)
        row = self.store.get_file("doomed.txt")
        self.assertEqual(row["state"], "deleted")
        self.assertFalse(quarantined_path.exists())

    def test_pinned_file_never_decays(self):
        cfg_path = config.config_path(self.root)
        cfg_path.write_text(
            "[thresholds]\n"
            "compress_after_days = 1\n"
            "archive_after_days = 3\n"
            "quarantine_after_days = 5\n"
            "delete_after_days = 1\n"
            "[scan]\n"
            'ignore = [".git", ".decay"]\n'
            'pinned = ["important.txt"]\n'
        )
        self.cfg = config.load_config(self.root)
        self._write("important.txt", "never touch me", age_days=100)
        engine.run_pass(self.cfg, self.store, apply=True)
        self.assertIsNone(self.store.get_file("important.txt"))
        self.assertTrue((self.root / "important.txt").exists())


class UndoTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        config.init_config(self.root)
        cfg_path = config.config_path(self.root)
        cfg_path.write_text(
            "[thresholds]\n"
            "compress_after_days = 1\n"
            "archive_after_days = 3\n"
            "quarantine_after_days = 5\n"
            "delete_after_days = 1\n"
            "[scan]\n"
            'ignore = [".git", ".decay"]\n'
            "pinned = []\n"
        )
        self.cfg = config.load_config(self.root)
        self.store = Store(self.cfg.db_path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _write(self, rel: str, content: str, age_days: float = 0):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if age_days:
            ts = time.time() - age_days * DAY
            os.utime(path, (ts, ts))
        return path

    def test_undo_restores_exact_content_from_compressed(self):
        self._write("doc.txt", "original bytes, please come back", age_days=2)
        engine.run_pass(self.cfg, self.store, apply=True)
        engine.run_pass(self.cfg, self.store, apply=True)
        row = self.store.get_file("doc.txt")
        self.assertEqual(row["state"], "compressed")

        restored_path = actions.restore(self.cfg, self.store, row)
        self.assertEqual(restored_path.read_text(), "original bytes, please come back")
        self.assertEqual(self.store.get_file("doc.txt")["state"], "active")

    def test_undo_restores_exact_content_from_quarantine(self):
        self._write("doc2.txt", "quarantined but not forgotten", age_days=10)
        engine.run_pass(self.cfg, self.store, apply=True)
        engine.run_pass(self.cfg, self.store, apply=True)
        row = self.store.get_file("doc2.txt")
        self.assertEqual(row["state"], "quarantined")

        restored_path = actions.restore(self.cfg, self.store, row)
        self.assertEqual(restored_path.read_text(), "quarantined but not forgotten")
        self.assertEqual(self.store.get_file("doc2.txt")["state"], "active")

    def test_cannot_undo_a_permanent_delete(self):
        self._write("doc3.txt", "irrelevant", age_days=10)
        engine.run_pass(self.cfg, self.store, apply=True)
        engine.run_pass(self.cfg, self.store, apply=True)
        row = self.store.get_file("doc3.txt")
        self.store.conn.execute(
            "UPDATE files SET state_changed_at = ? WHERE id = ?",
            (time.time() - 2 * DAY, row["id"]),
        )
        self.store.conn.commit()
        engine.run_pass(self.cfg, self.store, apply=True)

        row = self.store.get_file("doc3.txt")
        self.assertEqual(row["state"], "deleted")
        with self.assertRaises(actions.ActionError):
            actions.restore(self.cfg, self.store, row)

    def test_undo_on_active_file_is_an_error(self):
        self._write("live.txt", "still fresh", age_days=0)
        engine.run_pass(self.cfg, self.store, apply=True)
        row = self.store.get_file("live.txt")
        with self.assertRaises(actions.ActionError):
            actions.restore(self.cfg, self.store, row)


if __name__ == "__main__":
    unittest.main()
