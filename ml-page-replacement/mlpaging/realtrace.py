"""Real page-reference traces: parsers for common trace formats and
recorders that capture the references of a real program on this machine.

Every parser yields page numbers. ``build`` relabels them densely, drops
immediate repeats (always hits under every policy) and stops at a cap.

Formats
  lackey   Valgrind ``--tool=lackey --trace-mem=yes`` output (memory accesses)
  strace   ``strace -f -e trace=openat,open,read,pread64,lseek,close`` output (file reads)
  procmon  Windows Process Monitor CSV export (ReadFile/WriteFile with offsets)
  blkparse Linux blktrace text output (block I/O)
  msr      MSR Cambridge / SNIA IOTTA block traces (CSV, optionally .gz)
  plain    one address or page number per line (decimal or 0x hex)
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np

PAGE = 4096
FORMATS = ["auto", "lackey", "strace", "procmon", "blkparse", "msr", "plain"]


@dataclass
class Built:
    pages: np.ndarray
    raw_refs: int          # references parsed before dropping immediate repeats
    capped: bool


def build(page_iter, cap: int = 2_000_000, progress=None) -> Built:
    """Collapse immediate repeats, relabel densely, stop after ``cap`` references."""
    ids: dict = {}
    out = []
    last = None
    raw = 0
    capped = False
    for p in page_iter:
        raw += 1
        if p == last:
            continue
        last = p
        i = ids.get(p)
        if i is None:
            i = ids[p] = len(ids)
        out.append(i)
        if progress is not None and len(out) % 200_000 == 0:
            progress(len(out))
        if len(out) >= cap:
            capped = True
            break
    return Built(np.array(out, dtype=np.int64), raw, capped)


def _open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace", newline="")
    return open(path, "rt", errors="replace", newline="")


def sniff(path: str) -> str:
    """Guess the format from the first lines of the file."""
    with _open_text(path) as f:
        head = [f.readline() for _ in range(40)]
    text = "".join(head)
    if "Lackey" in text or re.search(r"^ ?[ILSM] +[0-9a-fA-F]+,\d+", text, re.M):
        return "lackey"
    if re.search(r"^(\d+ +)?(\[pid +\d+\] )?(openat|open|read|pread64|lseek|close|execve|mmap)\(", text, re.M):
        return "strace"
    if '"Operation"' in text or "Process Name" in text or re.search(r'"(ReadFile|WriteFile)"', text):
        return "procmon"
    if re.search(r"^\s*\d+,\d+\s+\d+\s+\d+\s+[\d.]+\s+\d+\s+[A-Z]\s+\S+\s+\d+ \+ \d+", text, re.M):
        return "blkparse"
    if re.search(r"^\d+,[^,]*,\d+,(Read|Write),\d+,\d+", text, re.M | re.I):
        return "msr"
    return "plain"


# ------------------------------------------------------------------ parsers

LACKEY = re.compile(r"^ ?([ILSM]) +([0-9a-fA-F]+),(\d+)")


def parse_lackey(lines, instructions: bool = False, page_size: int = PAGE):
    for line in lines:
        m = LACKEY.match(line)
        if not m:
            continue
        kind = m.group(1)
        if kind == "I" and not instructions:
            continue
        addr, size = int(m.group(2), 16), int(m.group(3))
        first, last = addr // page_size, (addr + max(size, 1) - 1) // page_size
        yield first
        if last != first:
            yield last


STRACE = re.compile(r"^(?:(\d+) +)?(?:\[pid +(\d+)\] )?(\w+)\((.*)\) += (-?\d+)")


def parse_strace(lines, page_size: int = PAGE):
    """File reads/writes as (file, page) references, tracking fd offsets."""
    files: dict = {}           # (pid, fd) -> [path, offset]
    for line in lines:
        m = STRACE.match(line)
        if not m:
            continue
        pid = m.group(1) or m.group(2) or "0"
        call, args, ret = m.group(3), m.group(4), int(m.group(5))
        if call in ("openat", "open") and ret >= 0:
            q = re.search(r'"((?:[^"\\]|\\.)*)"', args)
            files[(pid, ret)] = [q.group(1) if q else f"fd{ret}", 0]
            continue
        fd_m = re.match(r"\s*(\d+)", args)
        if not fd_m:
            continue
        key = (pid, int(fd_m.group(1)))
        if call == "close":
            files.pop(key, None)
            continue
        ent = files.get(key)
        if ent is None:
            continue
        if call == "lseek" and ret >= 0:
            ent[1] = ret
        elif call in ("read", "write") and ret > 0:
            yield from ((ent[0], p) for p in range(ent[1] // page_size, (ent[1] + ret - 1) // page_size + 1))
            ent[1] += ret
        elif call in ("pread64", "pwrite64") and ret > 0:
            off = int(args.rsplit(",", 1)[1])
            yield from ((ent[0], p) for p in range(off // page_size, (off + ret - 1) // page_size + 1))


PROCMON_DETAIL = re.compile(r"Offset:\s*([\d,]+),\s*Length:\s*([\d,]+)")


def parse_procmon(lines, page_size: int = PAGE, include_writes: bool = True):
    reader = csv.reader(lines)
    header = next(reader, None)
    if not header:
        return
    col = {name.strip().lstrip("﻿"): i for i, name in enumerate(header)}
    op_i, path_i, det_i = col.get("Operation"), col.get("Path"), col.get("Detail")
    if op_i is None or det_i is None:
        return
    ops = {"ReadFile"} | ({"WriteFile"} if include_writes else set())
    for row in reader:
        if len(row) <= max(op_i, det_i) or row[op_i] not in ops:
            continue
        m = PROCMON_DETAIL.search(row[det_i])
        if not m:
            continue
        off, ln = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
        path = row[path_i] if path_i is not None else ""
        for p in range(off // page_size, (off + max(ln, 1) - 1) // page_size + 1):
            yield (path, p)


BLK = re.compile(r"^\s*(\d+,\d+)\s+\d+\s+\d+\s+[\d.]+\s+\d+\s+([A-Z])\s+(\S+)\s+(\d+) \+ (\d+)")


def parse_blkparse(lines, page_size: int = PAGE, action: str = "Q"):
    for line in lines:
        m = BLK.match(line)
        if not m or m.group(2) != action:
            continue
        sector, n = int(m.group(4)), int(m.group(5))
        start, end = sector * 512, (sector + max(n, 1)) * 512 - 1
        for p in range(start // page_size, end // page_size + 1):
            yield (m.group(1), p)


def parse_msr(lines, page_size: int = PAGE, reads_only: bool = False):
    for row in csv.reader(lines):
        if len(row) < 6 or not row[4].isdigit():
            continue
        if reads_only and row[3].strip().lower() != "read":
            continue
        off, size = int(row[4]), int(row[5])
        for p in range(off // page_size, (off + max(size, 1) - 1) // page_size + 1):
            yield (row[2], p)


def parse_plain(lines, unit: str = "address", page_size: int = PAGE):
    for line in lines:
        s = line.strip().split(",")[0].split()[0] if line.strip() else ""
        if not s or s.startswith("#"):
            continue
        try:
            v = int(s, 16) if s.lower().startswith("0x") else int(s)
        except ValueError:
            continue
        yield v // page_size if unit == "address" else v


def load(path: str, fmt: str = "auto", cap: int = 2_000_000, instructions: bool = False,
         unit: str = "address", progress=None) -> tuple[Built, str]:
    """Parse a trace file of any supported format into page references."""
    if fmt == "auto":
        fmt = sniff(path)
    with _open_text(path) as f:
        if fmt == "lackey":
            it = parse_lackey(f, instructions)
        elif fmt == "strace":
            it = parse_strace(f)
        elif fmt == "procmon":
            it = parse_procmon(f)
        elif fmt == "blkparse":
            it = parse_blkparse(f)
        elif fmt == "msr":
            it = parse_msr(f)
        elif fmt == "plain":
            it = parse_plain(f, unit)
        else:
            raise ValueError(f"unknown format {fmt!r}")
        built = build(it, cap, progress)
    if built.pages.size == 0:
        raise ValueError(f"no page references found (read as {fmt})")
    return built, fmt


# ---------------------------------------------------------------- recorders

def tools() -> dict:
    return {"valgrind": shutil.which("valgrind") is not None,
            "strace": shutil.which("strace") is not None}


def record(argv: list[str], tool: str = "valgrind", cap: int = 2_000_000,
           instructions: bool = False, cwd: str | None = None, progress=None,
           timeout: float = 600) -> tuple[Built, dict]:
    """Run ``argv`` and capture its page references while it runs.

    valgrind: every load/store (and optionally instruction fetch) of the
    program, via the Lackey tool. strace: the program's file reads and
    writes at page granularity. Recording stops early at ``cap``.
    """
    if not argv:
        raise ValueError("empty command")
    if not tools().get(tool):
        raise RuntimeError(f"{tool} is not installed on this machine")
    log = tempfile.NamedTemporaryFile("w+", suffix=f".{tool}", delete=False)
    log.close()
    if tool == "valgrind":
        cmd = ["valgrind", "--tool=lackey", "--trace-mem=yes", f"--log-file={log.name}"] + argv
    else:
        cmd = ["strace", "-f", "-e", "trace=openat,open,read,pread64,write,pwrite64,lseek,close",
               "-o", log.name] + argv
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if tool == "valgrind":
            # stream the log while the program runs and stop at the cap
            built = build(_follow(log.name, proc, timeout,
                                  lambda lines: parse_lackey(lines, instructions)), cap, progress)
        else:
            proc.wait(timeout=timeout)
            with open(log.name, errors="replace") as f:
                built = build(parse_strace(f), cap, progress)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        try:
            os.unlink(log.name)
        except OSError:
            pass
    info = {"command": " ".join(argv), "tool": tool, "exit_code": proc.returncode,
            "instructions": instructions}
    if built.pages.size == 0:
        raise RuntimeError("the program produced no page references")
    return built, info


def _follow(path, proc, timeout, parser):
    """Yield parsed references from a log file that another process is writing."""
    import time
    deadline = time.time() + timeout

    def lines():
        with open(path, errors="replace") as f:
            buf = ""
            while True:
                chunk = f.read(1 << 20)
                if chunk:
                    buf += chunk
                    *full, buf = buf.split("\n")
                    yield from full
                    continue
                if proc.poll() is not None:
                    rest = f.read()
                    if rest:
                        buf += rest
                        continue
                    if buf:
                        yield buf
                    return
                if time.time() > deadline:
                    return
                time.sleep(0.05)

    yield from parser(lines())


def text_lines(data: bytes):
    return io.StringIO(data.decode(errors="replace"))
