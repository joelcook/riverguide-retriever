import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("v", "bakeoff_verdict.py"); v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
engines = [p.stem[5:] for p in sorted(Path("runs").glob("bake-*.json"))]
print(f"{'engine':16} {'golden':>7} {'med@1':>6} {'p90@4':>6} {'unf@4':>5} {'q/m@4':>6} {'q/m@16':>7} {'p90@16':>7} {'MiB':>6}")
for e in engines:
    try:
        r = v.result(e)
    except FileNotFoundError as x:
        print(f"{e:16} incomplete ({x.filename})"); continue
    if not r["started"]:
        print(f"{e:16} did not start: {r['why'][-120:]}"); continue
    print(f"{e:16} {r['golden']:>3}/{r['golden_of']:<3} {r['median_1']:>6} {r['p90_4']:>6} {r['unfinished_4']:>5} {r['qpm_4']:>6} {r['qpm_16']:>7} {r['p90_16']:>7} {r['memory_mib']:>6}")
if "--raw16" in sys.argv:
    for e in engines:
        p = Path(f"runs/load-bake-{e}.json")
        if p.exists():
            lv = json.loads(p.read_text())["levels"][-1]
            print(e, {k: lv[k] for k in lv if not isinstance(lv[k], list)})
