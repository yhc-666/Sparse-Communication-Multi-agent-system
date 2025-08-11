# Solver Engine
- This project implements logic solver engines in which reasoning paths are retrievable.
- Can refer to `sample_data/` for input and output format.

## Project Structure

```text
├── src/                       
│   ├── symbolic_solvers/      # Logic solvers (Prover9, Z3, PyKE, etc.)
│   │   ├── fol_solver/
│   │   ├── z3_solver/
│   │   ├── pyke_solver/
│   │   └── csp_solver/
│   ├── logic_inference.py     # Entry point for reasoning engine
│   └── ...
├── scripts/
│   ├── run_inference.sh       # Convenience launcher
│   └── demo.ipynb             # Jupyter notebook for quick demo
├── sample_data/               # Example inputs/output
├── requirements.txt           
└── README.md
```

## Start

1. Install dependencies (Python 3.12 recommended):
```bash
pip install -r requirements.txt
```

2. Configure the Prover9 executable path as described below.

3. Run inference:

```bash
bash scripts/run_inference.sh
# or
python src/logic_inference.py
```

4. Can use `scripts/demo.ipynb` for quick demo (recommend to open in Google-colab)


## ！Important: PROVER9 Path Configuration

You need to configure the PROVER9 path based on your operating system in `src/symbolic_solvers/fol_solver/prover9_solver.py` (around line 20-21).

### For Different Operating Systems (strongly suggest linux):

**Linux Users:**
```python
os.environ['PROVER9'] = PROVER9_PATH  # 直接复用原repo prover9
```

**macOS Users:**
```python
os.environ['PROVER9'] = '/opt/homebrew/bin'  # macOS version installed via Homebrew

