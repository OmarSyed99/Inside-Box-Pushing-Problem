# Inside‑Box Pushing ⇥ QMIX

A lightweight Gym environment + PyTorch QMIX implementation where four agents (one purposely faulty) must push a box to a goal **and vote to eliminate** the dead‑weight teammate.

---

## 1. Repository structure

| Path / file                         | Purpose                                                                                        |
|------------------------------------|------------------------------------------------------------------------------------------------|
| `env/box_push_env.py`              | Inside‑Box‑Pushing Gym environment (10 × 10 grid, voting)                                      |
| `algorithms/qmix/`                 | QMIX learner, replay buffer, mixing & hyper‑networks                                           |
| `run.py`                           | **Single** training run (command‑line flags below)                                             |
| `batch_run`                        | Bash helper / SLURM‑friendly wrapper for *many* runs                                           |
| `visualize.py`                     | Parses `training.log`, writes CSVs & graphs to `results/`                                      |
| `results/`                         | Auto‑created: CSVs + PNG figures                                                               |
| `config.yaml`                      | Default hyper‑parameters                                                                       |
| `requirements.txt`                 | Exact Python dependencies                                                                      |
| `training.log`                     | Example 5 000‑run log used in the paper                                                        |

---

## 2. Prerequisites

* **Python ≥ 3.8**
* **PyTorch ≥ 1.13**
* gymnasium • numpy • pandas • matplotlib • seaborn • scipy • tqdm • pyyaml • pygame  

## 3. Installation

git clone https://github.com/<your‑handle>/inside‑box‑pushing.git
cd inside‑box‑pushing

### (optional) create venv
```bash
python -m venv venv && source venv/bin/activate
```
```bash
pip install -r requirements.txt
```

## 4. How to run
### 4.1  Single run
```bash
python run.py --model qmix --episodes 200 --curriculum
```
Single run flags:

--episodes	Training episodes for this run	200

--model	Which learner to use (qmix, random, ...)

--curriculum	Curriculum level (1‑5) controlling obstacle mass, etc.	

--baseline  Used as a baseline and does not implement faulty agent.

The script appends all logs to training.log and stores checkpoints in models/.

### 4.2 Batch run
The helper script simply calls **`python run.py`** multiple times.
```bash
chmod +x batch_run
./batch_run -n 5 -- --model qmix --episodes 200 --curriculum 3
```
Batch Flags:

-n 3	batch_run script	Run python run.py three times in a row (i.e., create three independent training runs).

--	     is a separator — everything after this point is passed straight to run.py unchanged.

--model qmix	run.py	Use the QMIX architecture instead of any other model the code might support.

--episodes 200	run.py	Train (or train + evaluate) for 200 episodes in this run.

--curriculum 2	run.py	Start with curriculum level 2 (e.g., an easier map or lighter box) before progressing to harder 

settings, depending on how the curriculum logic is implemented.

## 5. Visualize
```bash
python visualize.py
```
The following graphs are generated when you use the command:
pass_fail_bar.png	Grouped bar chart—task vs. elimination success

-confusion_matrix.png	2 × 2 outcome matrix

-avg_reward_over_runs.png	Per‑run evaluation reward + trend line

-rate_over_runs.png	Success & elimination rates (two stacked panels)

-sliding_window_rates.png	50‑run moving mean ± SD

-run_results.csv, run_rates.csv	Parsed per‑run statistics

## 6. Questions or Issues
For any issues or questions, please email omarsyed@cmail.carleton.ca


