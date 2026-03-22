import subprocess
import os

scripts = ["linear_regression.py", "logistic_regression.py", "decision_tree.py", "random_forest.py"]

for script in scripts:
    print(f"--- Running {script} ---")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:")
        print(result.stderr)
