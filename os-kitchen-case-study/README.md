# Kitchen Synchronization Model

Working model for the case study *Synchronization in a Smart Restaurant Kitchen
Order Management System: A Case Study on the Dining Philosophers Problem Using
Semaphores and Monitors* (Rajveer Singh Chopra, 24BCE0122, Module 4).

| File | What it is |
|---|---|
| `kitchen_model.html` | Interactive model for the presentation. Open it in any browser; it needs no install or internet (fonts fall back if offline). |
| `kitchen_threads.py` | The same strategies on real Python threads (`threading.Semaphore`, `threading.Condition`). Reproduces the Section 11–12 experiment. |
| `results_threads.txt` | One run of `python3 kitchen_threads.py bench --repeats 5 --strategies nosync naive ordered arbitrator monitor fair`. |

Chef *i* needs station *S<sub>i</sub>* and *S<sub>(i+1) mod N</sub>*, exactly as
in Listings 10.1 and 10.2.

## Strategies

| Key | Strategy | What it shows |
|---|---|---|
| `nosync` | Shared `station_free[]` flags, check-then-set, no lock | Race condition (Section 9.2): two chefs on one station |
| `naive` | Every chef takes its left station first | Deadlock: all four Coffman conditions hold (Section 9.3) |
| `ordered` | Lower-numbered station first (Listing 10.1) | Circular wait removed |
| `arbitrator` | Counting semaphore `manager = N − 1` (strategy 2 in Section 9.5) | Circular wait removed |
| `monitor` | Dijkstra/Hoare monitor with `state[]` and `self[i]` (Listing 10.2) | Hold-and-wait removed; starvation still possible |
| `fair` | Monitor plus FIFO tickets (the extension named in Section 15.2) | Starvation removed |

## Running it

```bash
# Browser model
open kitchen_model.html            # macOS; or double-click the file

# Real threads (Python 3.9+, no packages needed)
python3 kitchen_threads.py bench                       # Section 12 tables, N = 5, 10, 20
python3 kitchen_threads.py live naive                  # watch the kitchen freeze
python3 kitchen_threads.py live monitor --seconds 8    # deadlock-free run
python3 kitchen_threads.py live nosync                 # race condition counter goes up
```

`live` slows prepare and cook times 100× so you can follow the states; the
summary at the end prints unscaled milliseconds.

## Presentation script (about 6 minutes)

The browser model has seven numbered steps across the top. Each one loads the
right settings and starts the kitchen. The model opens on step 2, already
deadlocked, so the problem is on screen from the first second.

1. **Race condition.** Point at the red `RACE` stations and the
   *Mutual-exclusion violations* tile. In the code panel the chips sit on
   `station_free[a] = false`, the moment between the check and the set.
2. **Deadlock.** Press *Reset*, then *Step* a few times. Each chef takes its
   left station (it slides next to them), then blocks on its right one. When
   the last chef blocks, the arrows turn red, the banner names the cycle, and
   the Coffman panel shows all four conditions holding. The semaphore panel
   shows every `station[i]` at value 0 with one chef in its queue.
3. **Resource ordering.** Same code with one change. Point at chef 4's code:
   it asks for S0 before S4, so the cycle can't close. *Circular wait* turns
   to *Removed*.
4. **Arbitrator.** The dots in the middle are the manager's free permits.
   When N − 1 chefs are already reaching, the next hungry chef queues on
   `wait(manager)` instead of taking a station.
5. **Monitor.** Chips sit on `self[i].wait()`, never on a station wait. The
   *Hold and wait now* tile stays at 0: test() grants both stations or none.
6. **Starvation.** Chefs 0 and 4 are on a break; chefs 1 and 3 cook back to
   back with overlapping shifts. Chef 2's dish count stays at 0 and its
   longest wait keeps growing, even though the kitchen never deadlocks. This
   is the limitation in Section 15.2.
7. **FIFO tickets.** Same timing. Chef 2 now cooks, because chef 1 can't jump
   ahead of a neighbour who asked first.

Finish on the benchmark at the bottom (or run `python3 kitchen_threads.py
bench` live). It repeats the Section 11 experiment for N = 5, 10 and 20.

## Results on real threads

From `results_threads.txt` (5 runs per cell, averaged). Your exact numbers
will vary from run to run, as with any thread timing.

| Strategy | N = 5 | N = 10 | N = 20 |
|---|---|---|---|
| Naive | deadlocked 5/5 | deadlocked 5/5 | deadlocked 5/5 |
| Semaphore (ordering), avg wait | 3.96 ms | 4.34 ms | 4.80 ms |
| Semaphore (arbitrator), avg wait | 4.22 ms | 7.84 ms | 7.72 ms |
| Monitor, avg wait | 3.49 ms | 3.20 ms | 3.03 ms |
| Monitor + FIFO tickets, avg wait | 4.15 ms | 3.35 ms | 3.59 ms |
| Semaphore (ordering), fairness σ | 0.68 ms | 0.96 ms | 1.04 ms |
| Monitor, fairness σ | 0.46 ms | 0.46 ms | 0.47 ms |

The pattern matches Section 12: the naive kitchen always deadlocks, the
monitor's wait stays flat as N grows, and the ordered semaphores' wait and
spread grow with N.

## Questions you may get

- **Why does ordering prevent deadlock?** A cycle of waiting chefs would need
  every chef to hold a lower-numbered station while waiting for a higher one.
  Chef N − 1 waits for S0 *before* taking S(N − 1), so the chain breaks there.
- **Why N − 1 permits?** With N stations shared by at most N − 1 chefs, by the
  pigeonhole principle at least one chef can get both of its stations.
- **Which Coffman condition does the monitor break?** Hold and wait: a chef
  gets both stations in one atomic step inside the monitor, or waits holding
  nothing.
- **Is the monitor starvation-free?** No. Step 6 shows a schedule where a chef
  waits forever. FIFO tickets (step 7) fix it.
- **`if` or `while` around `self[i].wait()`?** Listing 10.2 uses `if`
  (Hoare semantics). Python and Java signal-and-continue (Mesa), so the code
  uses `while`. Here both behave the same, because only chef *i* can change
  `state[i]` away from COOKING.
- **How is deadlock detected?** The model builds the wait-for graph (chef →
  holder of the station it waits for) and reports a cycle. The Python script
  uses the report's method: a 3 s acquisition timeout.
