"""Train the chosen model and write its weights as JSON for the web demo."""

from __future__ import annotations

import json
import sys

from . import experiment as E
from . import traces as T
from .learned import FEATURES, LogReg


def main(path="web/model.json", horizon=2.0, procs=2):
    train = T.workload_set("train")
    for t in train:
        E._TRACES[t.name] = t
    X, D = E.build_dataset(train, procs, E.FRACTIONS)
    m = LogReg().fit(X, (D > horizon).astype(float)).compile()
    out = {"kind": "logreg", "horizon": horizon, "features": FEATURES,
           "weights": m.w, "bias": m.b, "rows": int(len(D))}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:2])
