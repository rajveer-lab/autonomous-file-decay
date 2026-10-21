#!/usr/bin/env python3
"""Smart-kitchen order system as the Dining Philosophers Problem, with real threads.

Every chef is a ``threading.Thread`` and every cooking station is a binary
semaphore (``threading.Semaphore(1)``), following Section 10 of the case
study. Chef ``i`` needs station ``i`` and station ``(i + 1) % N``.

Strategies
  nosync      shared "free" flags, check-then-set with no lock  -> race condition
  naive       every chef takes its own (left) station first      -> deadlock
  ordered     lower-numbered station first (Listing 10.1)
  arbitrator  counting semaphore lets at most N-1 chefs reach for stations
  monitor     Dijkstra/Hoare monitor with condition variables (Listing 10.2)
  fair        monitor + FIFO tickets: a chef never overtakes a hungry
              neighbour that asked first (starvation-free extension, Sec. 15.2)

Usage
  python3 kitchen_threads.py bench                      # Section 12 tables
  python3 kitchen_threads.py bench --repeats 10 --json results.json
  python3 kitchen_threads.py live naive                 # watch it deadlock
  python3 kitchen_threads.py live monitor -n 5 --seconds 8
  python3 kitchen_threads.py live monitor --greedy 1 3  # starvation risk
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time

STRATEGIES = ("nosync", "naive", "ordered", "arbitrator", "monitor", "fair")
LABEL = {
    "nosync": "No synchronization",
    "naive": "Naive (left first)",
    "ordered": "Semaphore (ordering)",
    "arbitrator": "Semaphore (arbitrator)",
    "monitor": "Monitor",
    "fair": "Monitor + FIFO tickets",
}

PREPARING, HUNGRY, COOKING = "PREP", "WAIT", "COOK"


class Deadlock(Exception):
    """Raised when a chef has waited longer than the detection timeout."""


class Kitchen:
    """Instrumentation shared by all chefs (not part of the synchronization).

    Records who stands at which station so mutual exclusion can be checked,
    and the hungry-to-cooking wait of every dish.
    """

    def __init__(self, n: int):
        self.n = n
        self._lock = threading.Lock()
        self.at = [set() for _ in range(n)]
        self.violations = 0
        self.state = [PREPARING] * n
        self.waits: list[list[float]] = [[] for _ in range(n)]
        self.served = [0] * n
        self.deadlock_info = ""

    def enter(self, chef: int, s: int) -> None:
        with self._lock:
            self.at[s].add(chef)
            if len(self.at[s]) > 1:
                self.violations += 1

    def leave(self, chef: int, s: int) -> None:
        with self._lock:
            self.at[s].discard(chef)


# ----------------------------------------------------------------- strategies

class NoSync:
    """Check-then-set on shared flags: the race condition of Section 9.2.

    A chef spins until both of its stations look free, then marks them busy.
    The check and the set are separate steps, so two neighbours can both pass
    the check before either sets the flag, and both cook on the shared station.
    """

    def __init__(self, n, k, stop, timeout, gap):
        self.n, self.k, self.stop, self.timeout, self.gap = n, k, stop, timeout, gap
        self.free = [True] * n

    def pickup(self, i, rng):
        a, b = i, (i + 1) % self.n
        deadline = time.perf_counter() + self.timeout
        while not (self.free[a] and self.free[b]):   # busy-wait
            if self.stop.is_set():
                raise Deadlock
            if time.perf_counter() > deadline:
                raise Deadlock(f"C{i} spun for {self.timeout:.0f} s")
            time.sleep(0.0001)
        time.sleep(self.gap / 1000)                  # another chef may run here...
        self.free[a] = False                         # ...so this is not atomic
        self.free[b] = False
        self.k.enter(i, a)
        self.k.enter(i, b)

    def putdown(self, i):
        for s in ((i + 1) % self.n, i):
            self.k.leave(i, s)
            self.free[s] = True


class Semaphores:
    """naive / ordered / arbitrator: one binary semaphore per station."""

    def __init__(self, n, k, stop, timeout, pause, mode):
        self.n, self.k, self.stop, self.timeout, self.pause = n, k, stop, timeout, pause
        self.mode = mode
        self.station = [threading.Semaphore(1) for _ in range(n)]
        self.manager = threading.Semaphore(n - 1) if mode == "arbitrator" else None

    def _order(self, i):
        a, b = i, (i + 1) % self.n
        if self.mode == "ordered":
            return min(a, b), max(a, b)
        return a, b

    def _wait(self, sem, what):
        deadline = time.perf_counter() + self.timeout
        while not sem.acquire(timeout=0.05):
            if self.stop.is_set():
                raise Deadlock
            if time.perf_counter() > deadline:
                raise Deadlock(what)

    def pickup(self, i, rng):
        first, second = self._order(i)
        if self.manager is not None:
            self._wait(self.manager, f"C{i} waited on manager")
        self._wait(self.station[first], f"C{i} waited on S{first}")
        self.k.enter(i, first)
        if self.pause:
            time.sleep(rng.uniform(*self.pause) / 1000)
        self._wait(self.station[second],
                   f"C{i} holds S{first} and waited {self.timeout:.0f} s for S{second}")
        self.k.enter(i, second)

    def putdown(self, i):
        first, second = self._order(i)
        self.k.leave(i, second)
        self.station[second].release()
        self.k.leave(i, first)
        self.station[first].release()
        if self.manager is not None:
            self.manager.release()


class Monitor:
    """KitchenManager monitor (Listing 10.2), optionally with FIFO tickets.

    Python's Condition has signal-and-continue (Mesa) semantics, so the wait
    sits in a ``while`` loop instead of the listing's ``if``.
    """

    def __init__(self, n, k, stop, fair):
        self.n, self.k, self.stop, self.fair = n, k, stop, fair
        self.lock = threading.Lock()
        self.self_ = [threading.Condition(self.lock) for _ in range(n)]
        self.state = [PREPARING] * n
        self.ticket = [0] * n
        self.next_ticket = 0

    def _ahead(self, j, i):
        # neighbour j is hungry and asked before chef i
        return self.state[j] == HUNGRY and self.ticket[j] < self.ticket[i]

    def _test(self, i):
        left, right = (i + self.n - 1) % self.n, (i + 1) % self.n
        if (self.state[left] != COOKING and self.state[i] == HUNGRY
                and self.state[right] != COOKING
                and not (self.fair and (self._ahead(left, i) or self._ahead(right, i)))):
            self.state[i] = COOKING
            self.self_[i].notify()

    def pickup(self, i, rng):
        with self.lock:
            self.state[i] = HUNGRY
            self.ticket[i] = self.next_ticket
            self.next_ticket += 1
            self._test(i)
            while self.state[i] != COOKING:
                self.self_[i].wait(timeout=0.05)
                if self.stop.is_set():
                    raise Deadlock
        self.k.enter(i, i)
        self.k.enter(i, (i + 1) % self.n)

    def putdown(self, i):
        self.k.leave(i, (i + 1) % self.n)
        self.k.leave(i, i)
        with self.lock:
            self.state[i] = PREPARING
            self._test((i + self.n - 1) % self.n)
            self._test((i + 1) % self.n)


def make(strategy, n, k, stop, timeout, pause, gap=0.0):
    if strategy == "nosync":
        return NoSync(n, k, stop, timeout, gap)
    if strategy in ("naive", "ordered", "arbitrator"):
        return Semaphores(n, k, stop, timeout, pause, strategy)
    return Monitor(n, k, stop, fair=(strategy == "fair"))


# ---------------------------------------------------------------------- chefs

def chef(i, impl, k, barrier, stop, cycles, prep, cook, rng):
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        return
    done = 0
    while (cycles is None or done < cycles) and not stop.is_set():
        k.state[i] = PREPARING
        time.sleep(rng.uniform(*prep) / 1000)          # prepare_ingredients(i)
        k.state[i] = HUNGRY
        t0 = time.perf_counter()
        try:
            impl.pickup(i, rng)
        except Deadlock as e:
            if str(e) and not k.deadlock_info:
                k.deadlock_info = str(e)
            stop.set()
            return
        k.waits[i].append((time.perf_counter() - t0) * 1000)
        k.state[i] = COOKING
        time.sleep(rng.uniform(*cook) / 1000)          # cook_dish(i)
        k.served[i] += 1
        done += 1
        impl.putdown(i)
    k.state[i] = PREPARING


def run(strategy, n, cycles=15, prep=(1, 3), cook=(2, 5), pause=None,
        timeout=3.0, seed=0, greedy=(), duration=None, on_tick=None, gap=0.0):
    """Run one configuration and return its metrics.

    ``pause`` (ms) is the gap between picking up the first and the second
    station; by default only the naive chefs pause (3-6 ms), matching the
    methodology in Section 11. ``gap`` (ms) is how long a no-sync chef takes
    between seeing a station free and marking it busy.
    """
    if pause is None:
        pause = (3, 6) if strategy == "naive" else None
    k = Kitchen(n)
    stop = threading.Event()
    impl = make(strategy, n, k, stop, timeout, pause, gap)
    barrier = threading.Barrier(n)
    threads = []
    for i in range(n):
        p = (0, 0.05) if i in greedy else prep
        rng = random.Random(seed * 7919 + i)
        threads.append(threading.Thread(
            target=chef, args=(i, impl, k, barrier, stop, cycles, p, cook, rng),
            daemon=True))
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    if duration is not None:
        end = t0 + duration
        while time.perf_counter() < end and not stop.is_set():
            time.sleep(0.2)
            if on_tick:
                on_tick(time.perf_counter() - t0, k, impl)
        stop.set()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    per_chef = [statistics.fmean(w) for w in k.waits if w]
    all_waits = [x for w in k.waits for x in w]
    return {
        "strategy": strategy,
        "n": n,
        "deadlocked": bool(k.deadlock_info) or (duration is None and any(
            s < (cycles or 0) for s in k.served)),
        "deadlock_info": k.deadlock_info,
        "elapsed_s": elapsed,
        "dishes": sum(k.served),
        "served": list(k.served),
        "avg_wait_ms": statistics.fmean(all_waits) if all_waits else float("nan"),
        "max_wait_ms": max(all_waits) if all_waits else float("nan"),
        "fairness_sd_ms": statistics.pstdev(per_chef) if len(per_chef) > 1 else float("nan"),
        "violations": k.violations,
    }


# ------------------------------------------------------------------------ CLI

def bench(args):
    rows = []
    for n in args.n:
        for s in args.strategies:
            runs = [run(s, n, cycles=args.cycles, timeout=args.timeout, seed=r)
                    for r in range(args.repeats)]
            ok = [r for r in runs if not r["deadlocked"]]

            def mean(key, pool=ok):
                vals = [r[key] for r in pool if r[key] == r[key]]
                return statistics.fmean(vals) if vals else float("nan")

            row = {
                "strategy": s, "n": n, "runs": len(runs),
                "deadlocks": sum(r["deadlocked"] for r in runs),
                "violations": sum(r["violations"] for r in runs),
                "elapsed_s": mean("elapsed_s", runs),
                "avg_wait_ms": mean("avg_wait_ms"),
                "fairness_sd_ms": mean("fairness_sd_ms"),
                "max_wait_ms": mean("max_wait_ms"),
            }
            rows.append(row)
            print(f"  ran {LABEL[s]:<24} N={n:<3} "
                  f"deadlocks {row['deadlocks']}/{row['runs']}", file=sys.stderr)

    def fmt(x, unit=""):
        return "-" if x != x else f"{x:.2f}{unit}"

    print(f"\nReal-thread benchmark: {args.cycles} dishes per chef, prepare 1-3 ms, "
          f"cook 2-5 ms, {args.repeats} run(s) per cell, deadlock timeout "
          f"{args.timeout:g} s.\nWait and fairness are averaged over runs that "
          "finished; '-' means every run deadlocked.\n")
    head = (f"{'Strategy':<24}{'N':>4}{'Deadlocked':>12}{'ME violations':>15}"
            f"{'Elapsed':>10}{'Avg wait':>11}{'Fairness sd':>13}{'Max wait':>11}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{LABEL[r['strategy']]:<24}{r['n']:>4}"
              f"{str(r['deadlocks']) + '/' + str(r['runs']):>12}{r['violations']:>15}"
              f"{fmt(r['elapsed_s'], ' s'):>10}{fmt(r['avg_wait_ms'], ' ms'):>11}"
              f"{fmt(r['fairness_sd_ms'], ' ms'):>13}{fmt(r['max_wait_ms'], ' ms'):>11}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")


COLORS = {PREPARING: "\x1b[90m", HUNGRY: "\x1b[33m", COOKING: "\x1b[32m"}


def live(args):
    scale = args.scale
    print(f"{LABEL[args.strategy]} with N={args.n} chefs; times slowed x{scale:g} "
          "so you can follow them. Ctrl+C to stop.")
    print("PREP = preparing ingredients, WAIT = hungry (waiting for stations), "
          "COOK = cooking (critical section)\n")

    def tick(t, k, impl):
        cells = []
        for i in range(k.n):
            st = k.state[i]
            c = COLORS[st] if sys.stdout.isatty() else ""
            r = "\x1b[0m" if c else ""
            cells.append(f"{c}C{i} {st}{r}")
        print(f"t={t:5.1f}s  dishes={sum(k.served):4d}  "
              f"ME violations={k.violations:<3d} | " + "  ".join(cells))

    res = run(args.strategy, args.n, cycles=None,
              prep=(1 * scale, 3 * scale), cook=(2 * scale, 5 * scale),
              pause=((3 * scale, 6 * scale) if args.strategy == "naive"
                     else None),
              timeout=args.timeout, seed=args.seed, greedy=set(args.greedy),
              duration=args.seconds, on_tick=tick, gap=1.0 * scale)
    print()
    if res["deadlock_info"]:
        print(f"DEADLOCK detected: {res['deadlock_info']}.")
        print("Every chef holds one station and waits for its neighbour's: "
              "circular wait.")
    print(f"dishes per chef: {res['served']}")
    if res["dishes"]:
        print(f"average wait {res['avg_wait_ms'] / scale:.2f} ms (unscaled), "
              f"fairness sd {res['fairness_sd_ms'] / scale:.2f} ms, "
              f"longest wait {res['max_wait_ms'] / scale:.2f} ms")
    print(f"mutual-exclusion violations (two chefs on one station): {res['violations']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="reproduce the Section 12 experiment")
    b.add_argument("-n", type=int, nargs="+", default=[5, 10, 20])
    b.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                   default=["naive", "ordered", "arbitrator", "monitor", "fair"])
    b.add_argument("--cycles", type=int, default=15)
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--timeout", type=float, default=3.0)
    b.add_argument("--json")
    b.set_defaults(func=bench)

    lv = sub.add_parser("live", help="watch one strategy run")
    lv.add_argument("strategy", choices=STRATEGIES)
    lv.add_argument("-n", type=int, default=5)
    lv.add_argument("--seconds", type=float, default=8)
    lv.add_argument("--scale", type=float, default=100,
                    help="slow-down factor for prepare/cook times (default 100)")
    lv.add_argument("--timeout", type=float, default=3.0)
    lv.add_argument("--greedy", type=int, nargs="*", default=[],
                    help="chefs that skip preparation (starvation scenario)")
    lv.add_argument("--seed", type=int, default=0)
    lv.set_defaults(func=live)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
