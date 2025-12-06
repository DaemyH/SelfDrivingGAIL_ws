# Self-Driving-GAIL

In this project, an end-to-end autonomous driving system is implemented. A deep learning model is trained to imitate an expert driver's behavior using Generative Adversarial Imitation Learning (GAIL) algorithm [[1]](#1). The project is developed with [PyTorch](https://pytorch.org/) and [CARLA Simulator](https://carla.org/).

This is the final project for my BSc in Computer Engineering at Amirkabir University of Technology (AUT), September 2022.

## System's Inputs and Outputs

The system works in and end-to-end manner [[2]](#2). At each moment, the model takes as inputs the images of three RGB cameras, a high-level navigational command (that instructs the vehicle to "turn left", "turn right", or "go straight") [[3]](#3), and the current speed of the vehicle. Based on these inputs, the model directly produces three control signals, namely, Throttle, Steer, and Brake, which are used to drive the car.

<div align="center">
    <img src="figures/blackbox.jpg" width="500" alt="blackbox">
</div>

## Camera Setup

The camera setup is as follows: There is one front camera and two wide-angle cameras on the left and right sides of the vehicle.

Left Camera             |  Front Camera        |  Right Camera
:-------------------------:|:-------------------------:|:-------------------------:
![Left Camera](figures/left-camera.png)  |  ![Front Camera](figures/front-camera.png) | ![Right Camera](figures/right-camera.png)


## System Architecture

The architecure is based on the Reinforcement Learning loop. The agent learns to drive from interactions with the environment and also, from the expert driving dataset. The Actor-Critic algorithm, PPO [[4]](#4), is used to implement the agent. 

<div align="center">
    <img src="figures/system.jpg" width="500" alt="system">
</div>

Three learning signals are used to train the model:
- **Generative Adversarial Imitation Learning**: The model learns from the rewards predicted by a Discriminator that tries to distinguish bewteen expert driving behavior and that of the agent [[5]](#5).
- **Behavioral Clonining**: The agent has direct access to the expert dataset and learns to imitate expert decisions through supervised learning [[6]](#6).
- **Explicit Rewards**: In order to help the model avoid obstacles and keep the vehicle in its lane, a negative reward is produced whenever a lane invasion occurs.

## Network Architecure

**Agent Network**: A shared network is used for the Actor and the Critic modules of the PPO agent. The model processes the camera images separately, and then fuses these sensory information to get a unified state vector. Then, based on the given high-level command, one head of the model is chosen to generate the control signals and the state value.

<br>

<div align="center">
    <img src="figures/actor-critic.jpg" width="700" alt="actor-critic">
</div>

<br>

**Discriminator Network**: This is similar to the Agent network, but takes the action as an input as well, and outputs the probability of the state-action pair belonging to the expert dataset.

<div align="center">
    <img src="figures/disc.jpg" width="700" alt="disc">
</div>

## Dataset

The training and testing is performed in Town 2 of CARLA simluator. The model is trained on intersections 1-6, and intersections 7-8 are used for testing the performance of the system on unseen routes.

<div align="center">
    <img src="figures/town-2.jpg" width="300" alt="town-2">
</div>

<br>
At each intersection, the expert data is gathered on two left turns and two right turns (in total, 24 turns). CARLA's open-source navigation agent and PID controller are used to generate the expert data automatically.

<br>
<br>
For simplicity, it is assumed that there's no traffic on the roads and traffic lights are not taken into consideration.

## Sample Results

**Long Route:** A simple route which includes roads the model has not seen during training. We can observe that the model has learned to follow high-level commands and drives safely.

https://user-images.githubusercontent.com/36497794/229384560-59c0bf97-97eb-4109-93a2-4fef26303c4e.mp4

<br>

**Noisy Steering Wheel:** In this experiment, the trained model is tested on a simulated vehicle whose steering wheel turns to right at random moments. As we can see, the system is able to react quickly and keep the vehicle in its lane. 

https://user-images.githubusercontent.com/36497794/229385027-5c74754b-a69c-4f39-a32e-6077b3b5d2bd.mp4

<br>
What's interesting is that the model has not seen this type of error in the expert dataset or in the training environment. But due to the inherent exploration present in the stochastic policy of PPO, and also, the explicit reward explained before, the model has learned by itself to handle such errors.

## How to Run (step by step)
1. **환경 준비**
   - CARLA 시뮬레이터 서버를 실행합니다(`--host 127.0.0.1 --port 2000`).
   - Python 의존성 설치: `pip install -r requirements.txt` (필요 시 CARLA egg 경로를 `PYTHONPATH`에 추가).
2. **전문가 데이터 수집**
   - CARLA가 켜진 상태에서 `python generate_data.py --num-episodes 4 --record True --autopilot True --no-screen True` 실행.
   - `expert_data/<timestamp>/` 폴더에 `expert_states.npy`, `expert_commands.npy`, `expert_speeds.npy`, `expert_actions.npy`, `len.txt`가 저장됩니다.
3. **Behavioral Cloning 사전학습**
   - `python train_bc.py --seed 1 --learning-rate 3e-4 --branched True --use-cuda True`
   - 체크포인트는 `models/bc/`에 저장되며 TensorBoard 로그는 `runs/` 폴더에 기록됩니다.
4. **BC+GAIL 통합 학습**
   - `python train_bc_gail.py --total_timesteps 150000 --ppo-learning-rate 1e-4 --disc-learning-rate 1e-4 --branched True --use-cuda True`
   - 5번째 업데이트마다 PPO/Discriminator 가 `models/bc_gail/`에 저장됩니다.
5. **NMPC 결합 평가**
   - CARLA 서버를 켠 뒤 `python evaluate_agent.py --agent-name bc_gail --on-test-set True --deterministic True --use-cuda True` 실행.
   - `algo.nmpc`가 정책 출력을 참조 궤적으로 변환하여 NMPC를 풀이하고, 실패 시 PID fallback을 적용합니다. 실행 로그에 경로 오차·제어 변화량·fallback 횟수가 출력됩니다.

## NMPC controller code locations

If you pulled the branch that contains the NMPC evaluation work, you should see the following files and imports in your local checkout:

- `algo/nmpc.py`: lightweight NMPC utilities (kinematic bicycle dynamics, constraints, PID fallback, waypoint helpers).
- `evaluate_agent.py`: imports `NMPCConfig`, `NMPCController`, and `PIDController` from `algo.nmpc` and wires them into evaluation.

If those files are missing after a fresh `git clone`, make sure you are on the updated branch (e.g., `work` or the PR branch) by running `git status` and `git branch --show-current`.

## NMPC controller code locations

If you pulled the branch that contains the NMPC evaluation work, you should see the following files and imports in your local checkout:

- `algo/nmpc.py`: lightweight NMPC utilities (kinematic bicycle dynamics, constraints, PID fallback, waypoint helpers).
- `evaluate_agent.py`: imports `NMPCConfig`, `NMPCController`, and `PIDController` from `algo.nmpc` and wires them into evaluation.

If those files are missing after a fresh `git clone`, make sure you are on the updated branch (e.g., `work` or the PR branch) by running `git status` and `git branch --show-current`.

## How to verify the NMPC files locally

If you want to double-check that the NMPC code and the PPO/NMPC wiring are present after cloning, follow the steps below.

1. **Clone the repo**
   ```bash
   git clone <REPO_URL>
   cd SelfDrivingGAIL_ws
   ```
2. **Confirm you are on the branch with NMPC support**
   ```bash
   git branch --show-current
   # expected: work (or the branch name that contains the NMPC changes)
   git status
   ```
   If you are on a different branch, switch to the correct one:
   ```bash
   git checkout work   # or the branch name shared with you
   ```
3. **Verify the NMPC files exist**
   ```bash
   ls algo/nmpc.py
   rg "NMPCController" evaluate_agent.py
   rg "PIDController" evaluate_agent.py
   ```
   You should see `algo/nmpc.py` and NMPC/PID imports in `evaluate_agent.py`.
4. **Optional: run a quick import check**
   ```bash
   python - <<'PY'
   from algo.nmpc import NMPCConfig, NMPCController, PIDController
   import evaluate_agent
   print('NMPC imports OK')
   PY
   ```
   This confirms Python can locate the NMPC helpers and the evaluation script. If this fails, re-check the branch and your Python environment.

## Compare policy-only, PID, and NMPC controllers

To build a clear comparison baseline for your paper, you can run the same trained PPO policy with three different control backends:

- **policy**: raw PPO outputs (throttle/steer/brake) applied directly.
- **pid**: PPO for lateral steer, PID for longitudinal speed tracking (uses the policy's target speed).
- **nmpc**: PPO short-horizon reference converted to NMPC, with PID fallback on solver failures.

Use the new `--control-mode` flag in `evaluate_agent.py` to switch among them and log per-route metrics for side-by-side analysis:

```bash
# policy-only baseline
python evaluate_agent.py --control-mode policy --results-path logs/policy_only.csv

# policy + PID (speed) baseline
python evaluate_agent.py --control-mode pid --results-path logs/policy_pid.csv

# policy + NMPC (default)
python evaluate_agent.py --control-mode nmpc --results-path logs/policy_nmpc.csv
```

Each CSV will contain per-route tracking error, control smoothing, and fallback counts so you can quantify the benefit of NMPC over the policy-only and PID-assisted baselines.

## Reference Papers
- <a id="1">[1]</a> [Generative Adversarial Imitation Learning, NIPS (2016)](https://arxiv.org/abs/1606.03476)
- <a id="2">[2]</a> [End to End Learning for Self-Driving Cars, arXiv (2017)](https://arxiv.org/abs/1604.07316)
- <a id="3">[3]</a> [End-to-end Driving via Conditional Imitation Learning, ICRA (2018)](https://arxiv.org/abs/1710.02410)
- <a id="4">[4]</a> [Proximal Policy Optimization Algorithms, arXiv (2017)](https://arxiv.org/abs/1707.06347)
- <a id="5">[5]</a> [Generative Adversarial Imitation Learning for End-to-End Autonomous Driving on Urban Environments, SSCI (2021)](https://arxiv.org/abs/2110.08586)
- <a id="6">[6]</a> [Augmenting GAIL with BC for sample efficient imitation learning, PMLR (2021)](https://arxiv.org/abs/2001.07798)
