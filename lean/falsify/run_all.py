"""Run every numeric falsification test; exit status 0 iff all pass."""
import pathlib
import subprocess
import sys

here = pathlib.Path(__file__).parent
failed = []
for script in sorted(here.glob("*.py")):
    if script.name.startswith("_") or script.name == "run_all.py":
        continue
    result = subprocess.run([sys.executable, str(script)], cwd=here)
    if result.returncode != 0:
        failed.append(script.name)
print("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}")
sys.exit(1 if failed else 0)
