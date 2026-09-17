import unittest
from pathlib import Path

from filedecay.scanner import is_ignored, is_pinned


class TestIsIgnored(unittest.TestCase):
    def test_ignores_dotdecay_anywhere_in_path(self):
        self.assertTrue(is_ignored(Path(".decay/archive/foo.txt"), [".decay"]))
        self.assertTrue(is_ignored(Path("sub/.decay/foo.txt"), [".decay"]))

    def test_ignores_matching_glob(self):
        self.assertTrue(is_ignored(Path("build/output.o"), ["*.o"]))
        self.assertFalse(is_ignored(Path("build/output.txt"), ["*.o"]))

    def test_unrelated_file_not_ignored(self):
        self.assertFalse(is_ignored(Path("notes/todo.md"), [".git", "node_modules"]))


class TestIsPinned(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(is_pinned(Path("important.txt"), ["important.txt"]))

    def test_glob_match(self):
        self.assertTrue(is_pinned(Path("keep/readme.md"), ["keep/*"]))

    def test_no_match(self):
        self.assertFalse(is_pinned(Path("scratch/junk.txt"), ["keep/*"]))


if __name__ == "__main__":
    unittest.main()
