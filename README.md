<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Dynamic Grouping-Guided Flow Matching (DyGFlow)</h1>
      <br>
      <h2>Dynamic Grouping-Guided Flow Matching for Offline Multi-Agent Reinforcement Learning</h2>
      <p>Based on <a href="https://arxiv.org/abs/2511.05005">MAC-Flow</a></p>
    </summary>
  </ul>
</div>
</div>

## Overview
DyGFlow extends MAC-Flow with dynamic agent grouping, group-guided hierarchical
attention, a group-aware mixer, and group-consistency regularization for offline
multi-agent reinforcement learning.

## Installation
```bash
# Environment (OG-MARL: https://github.com/instadeepai/og-marl)
conda create -n recipe python=3.8
conda activate recipe
python -m pip install --upgrade pip
pip install -r requirements/datasets.txt
```

Install the SMAC, MPE, MA-MuJoCo environment you plan to use:

```bash
# SMAC v1
bash install_environments/smacv1.sh
pip install -r install_environments/requirements/smacv1.txt

# SMAC v2
bash install_environments/smacv2.sh
pip install -r install_environments/requirements/smacv2.txt

# MPE
pip install -r install_environments/requirements/pettingzoo.txt

# MA-MuJoCo
bash install_environments/mujoco200.sh
pip install -r install_environments/requirements/mamujoco200.txt
```

If you need baseline TensorFlow dependencies:

```bash
pip install -r requirements/baselines.txt
```

## Quick start

```bash
# DyGFlow
python smac_main.py --agent_name dygflow --env smac_v1 --source og_marl --scenario 2s3z --dataset Good --seed 0 --agent.alpha 3.0
```
