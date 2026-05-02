# SO101 Lift Cube — Isaac Lab RL → 30fps 추론 → SmolVLA 데이터 생산 파이프라인

Isaac Sim 5.1 + Isaac Lab + [leisaac](https://github.com/LightwheelAI/leisaac) 위에서 **SO101 6-DOF 로봇팔**이 치약통 모양 cube를 집어 올리는 정책을 강화학습으로 학습하고, **30fps 추론**으로 다양한 spawn 조건에서 성공 에피소드를 대량 수집하여 **SmolVLA 파인튜닝**의 학습 데이터로 사용합니다.

> **결과 요약 (v41 SAC, decimation=3 = 33Hz 추론)**
> 5가지 다양한 cube 스폰 (위치 ±15 cm, yaw 0/45/-60/90/135°)에서 각 10초(333 step) 롤아웃, 80% 이상 hold 유지 시 SUCCESS로 저장. front/side/wrist 3-camera MP4 + npz frame + trajectory.npz (qpos, action, target_qpos, cube pose) 저장 → 그대로 LeRobot v2 dataset으로 변환 가능.

> **▶︎ 30fps 추론 성공 영상**: [`inference_results/v41_30fps/`](inference_results/v41_30fps/) 에 5개 SUCCESS 에피소드 (front/side/wrist 3-camera MP4 + trajectory.npz) 포함. 자세한 표는 [§5.6 — 실제 추론 결과](#56-실제-추론-결과-v41_30fps) 참고.

---

## 1. 파이프라인 개요

```
[Isaac Sim 5.1 + Isaac Lab]
        │
        ├── (A) PPO  — train_rl.py        (rsl_rl, 1-3 envs, 대시보드, 초기 데모)
        │
        └── (B) SAC fast — train_sac_fast.py (Squint-style, 2048 envs,
                                              C51 distributional critic,
                                              CudaGraph + bf16, 본 학습)
                │
                ▼
        체크포인트 best_ckpt.pt
                │
                ▼
[infer_sac.py — 30fps 추론]
   decimation = 3 → 100Hz / 3 ≈ 33Hz
   5 diverse spawn × 10초(333 steps)
   hold criteria: cube z ≥ 15 cm + grasped + pan err < 0.262 rad
   ≥ 80% hold ratio → SUCCESS 저장
                │
                ▼
   SUCCESS_ep{NN}_x..y..yaw..ret..
   ├─ trajectory.npz   (qpos, actions, cube_pos/quat, target_qpos)
   ├─ cam_front.npz    (uint8 RGB, 480x640x3)
   ├─ cam_side.npz
   ├─ cam_wrist.npz
   ├─ front.mp4 / side.mp4 / wrist.mp4   (30 fps H.264)
                │
                ▼
[convert_replay_to_lerobot.py]
   → /data/lerobot_dataset/sim_lift  (LeRobot v2 형식)
                │
                ▼
[SmolVLA 파인튜닝]   ← 이 repo의 범위 밖 (별도 repo)
```

---

## 2. 디렉토리 구조

```
leisaac-rl-so101/
├── Dockerfile               # nvcr.io/nvidia/isaac-sim:5.1.0 + torch 2.7+cu128 + leisaac
├── Dockerfile.smolvla       # SmolVLA 추론·변환용 컨테이너
├── docker-compose.yml       # leisaac + smolvla 두 서비스 정의
├── .gitignore
└── delivery/                # 컨테이너 내부 /workspace/delivery 로 마운트되는 본체
    ├── README.md            # PPO 학습 상세 문서 (보상/벌점/커리큘럼/PPO 하이퍼파라미터)
    ├── setup_rl_env.sh      # 컨테이너 진입 후 1회 실행: leisaac override 적용 + xvfb + rsl_rl
    ├── auto_train.sh        # 학습이 죽어도 자동 재시작 (PPO)
    ├── run_rl_smoke.sh      # 200-iter PPO smoke test
    ├── run_rl_train.sh      # PPO 본 학습 launcher
    ├── configs/
    │   ├── reward_config.yaml   # 보상·벌점·커리큘럼·PPO 모든 하이퍼파라미터
    │   ├── replay_config.yaml
    │   └── robot_config.yaml    # SO101 + STS3215 모터 스펙 참조 문서
    ├── leisaac_overrides/        # leisaac 패키지 덮어쓰기 (setup_rl_env.sh가 자동 적용)
    │   ├── lerobot.py           # 모터 effort 3.5 Nm, velocity 6.3 rad/s (STS3215 실제 스펙)
    │   └── mdp/
    │       ├── rewards.py       # jaw 기준 + lateral_deviation/velocity_excess/object_out_of_reach
    │       └── terminations.py  # cube 이동 기반 out_of_reach
    └── packages/
        ├── common/
        │   ├── constants.py     # SO101 joint 이름, 모터 한계, STS3215 스펙
        │   └── robot_utils.py   # leisaac ↔ lerobot 액션 변환
        ├── sim/
        │   ├── train_rl.py             # PPO (rsl_rl, 대시보드 :8888, 초기 도입)
        │   ├── train_sac.py            # SAC v1 (state)
        │   ├── train_sac_fast.py       # SAC vmap + C51 + CudaGraph + bf16 (★ 본 학습)
        │   ├── train_sac_vision.py     # SAC + 카메라 입력 실험
        │   ├── pretrain_bc.py / pretrain_bc_v5.py  # 데모 → BC 사전학습
        │   ├── infer_sac.py            # ★ 30fps 추론 + 성공 에피소드 저장
        │   ├── debug_infer.py
        │   ├── replay_episodes.py      # 저장된 npz frames → MP4 + 웹 viewer (:8890)
        │   ├── replay_episode.py / replay_demos.py / replay_success.py / replay_top20.py
        │   ├── replay_trajectory.py
        │   ├── dashboard.py            # 학습 중 :8888 실시간 모니터링
        │   ├── episode_tracker.py      # 단계 분류 (grasp / lift{3..9}cm / hold)
        │   ├── data_saver.py           # frame npz + meta.json 저장
        │   ├── env_builder.py
        │   ├── iter_logger.py
        │   └── env_setup/
        │       ├── env_config.py             # configure_env, apply_motor_limits
        │       ├── maniskill_rewards.py      # _is_grasped, _fold_3joint 등 헬퍼
        │       ├── target_delta_action.py    # delta-action wrapper (target_qpos 추적)
        │       ├── curriculum.py             # mass / spawn range / penalty schedule
        │       ├── helpers.py
        │       ├── config.py
        │       └── rewards/                  # reach / align / grasp / lift / open / close / penalties
        └── vla/
            ├── convert_replay_to_lerobot.py  # ★ 성공 에피소드 → LeRobot v2 dataset
            └── serve_smolvla.py              # SmolVLA action server (sim 추론용 client)
```

---

## 3. 컨테이너 실행

### 3.1 호스트 사전 요구사항
- NVIDIA GPU + 최신 드라이버 (RTX 4090 24GB로 검증)
- Docker + nvidia-container-toolkit
- (옵션) 학습 모니터링용 8888/8890 포트 개방

### 3.2 빌드 & 기동

```bash
git clone <이 repo URL> leisaac-rl-so101
cd leisaac-rl-so101

# 컨테이너 내부에서 마운트할 경로들을 호스트에 미리 만들어 둡니다.
# (.gitignore에서 제외된 경로 — git 추적 안 됨)
mkdir -p cache/{kit,ov,pip,glcache,computecache,huggingface}
mkdir -p logs/{omniverse,kit}
mkdir -p data/{ov/Kit,rl_output,lerobot_dataset,smolvla_ckpt}
mkdir -p documents datasets

# Isaac Sim USD 에셋 (assets/robots/*.usd, assets/scenes/...) 은 별도 배포.
# 사내 잘 알려진 경로에서 받거나 leisaac 원본 또는 lekiwi/so101_follower USD를 복사하세요.
mkdir -p assets/robots assets/scenes

docker compose build leisaac
docker compose up -d leisaac
docker exec -it leisaac bash
```

### 3.3 컨테이너 안에서 1회 셋업

```bash
cd /workspace/delivery
./setup_rl_env.sh
```

`setup_rl_env.sh` 가 하는 일:
1. `leisaac_overrides/` 의 3개 파일을 leisaac 설치 경로 (`/workspace/leisaac/source/leisaac/leisaac/...`) 에 덮어쓰기
2. lift_cube/mdp/`__init__.py`에 `from .rewards import *` 추가
3. `__pycache__` 정리
4. `pyyaml`, `tensordict`, `tensorboard` 설치
5. xvfb 설치 + `:1` 디스플레이 기동
6. `rsl_rl` 클론 & `pip install -e .`
7. `train_rl.py` 의 env id 패치

---

## 4. 학습

이 repo에는 두 가지 학습 경로가 있습니다.

### 4.1 (A) PPO — `train_rl.py`

상세 문서: [`delivery/README.md`](delivery/README.md). 보상/벌점/커리큘럼/PPO 하이퍼파라미터 표가 모두 정리되어 있습니다.

```bash
# Smoke test (200 iter)
./run_rl_smoke.sh

# 본 학습
./run_rl_train.sh                                  # config 기본값
./run_rl_train.sh --num-envs 3 --max-iterations 1000000

# 자동 재시작 루프 (NaN/OOM 대비)
./auto_train.sh
```

**대시보드** : `http://<서버IP>:8888`
- 카메라 3개 실시간 스트리밍 (front, side, wrist)
- iteration / episode / reward 추이 (20-ep 이동평균 + 최고값)
- phase (reaching → grasping → lifting → holding)
- success rate
- 커리큘럼 단계 + 현재 cube mass

### 4.2 (B) SAC — `train_sac_fast.py` (★ 본 학습용)

[Squint](https://github.com/aalmuzairee/squint) 구현을 그대로 따라간 고속 SAC. PPO 대비 sample efficiency가 훨씬 높아 v41 이후 본 학습은 모두 SAC fast 로 진행했습니다.

핵심 디자인:
- **2048 vector envs** (RTX 4090 1장 기준)
- **C51 Distributional Critic** (101 atoms)
- **vmap Q-network 앙상블** (`from_modules`)
- **CudaGraphModule** + `torch.compile`
- **bfloat16 autocast**
- **bootstrap_at_done="always"** — done 시점에도 항상 부트스트랩
- **torchrl ReplayBuffer** + `LazyTensorStorage` + TensorDict
- **Delayed policy updates** (`policy_freq=4`)
- **Alpha update inside critic step** (Squint ordering)

```bash
# Foreground
/isaac-sim/python.sh packages/sim/train_sac_fast.py \
  --num-envs 2048 \
  --total-timesteps 100000000000 \
  --http-port 8888 \
  --save-dir /data/rl_sac_state_v45

# Background
docker exec -d leisaac bash -c '
  cd /workspace/delivery && \
  /isaac-sim/python.sh packages/sim/train_sac_fast.py \
    --num-envs 2048 \
    --total-timesteps 100000000000 \
    --http-port 8888 \
    --save-dir /data/rl_sac_state_v45 \
    > /data/sac_state.log 2>&1
'

# Resume
... --resume /data/rl_sac_state_v45/best_ckpt.pt
```

옵션:
| 옵션 | 기본 | 설명 |
|------|------|------|
| `--num-envs` | 2048 | 4090 24GB 기준 2048 안정 |
| `--total-timesteps` | 1e8 | 사실상 무한 (best_ckpt 갱신만 의미) |
| `--save-dir` | `/data/rl_sac_fast` | best_ckpt.pt + 주기 ckpt + log |
| `--resume` | "" | `.pt` 경로 |
| `--no-cudagraphs` | off | 디버깅용 |
| `--seed` | 1 | |
| `--http-port` | 8888 | dashboard.py 와 공유 |
| `--config` | `configs/reward_config.yaml` | 보상/벌점은 PPO 와 동일 config 공유 |

`train_sac_vision.py` 는 카메라를 observation에 포함시킨 실험 버전으로, 본 학습에는 사용하지 않았습니다 (state-only 가 안정적).

---

## 5. 30fps 추론 — `infer_sac.py` ★

학습이 끝난 정책에 대해 **decimation = 3** 으로 환경 simulation step rate를 1/3로 줄여 (100Hz → 33.3Hz) 30fps MP4를 그대로 저장하도록 한 추론 스크립트입니다. 실 로봇 30fps 카메라/제어 주기와 정렬되어, 이대로 SmolVLA 학습 데이터로 사용할 수 있습니다.

### 5.1 핵심 설정 (스크립트 상단에서 수정)

```python
DEVICE       = "cuda:0"
CKPT         = "/data/rl_sac_state_v41/best_ckpt.pt"
OUTPUT       = "/data/infer_v41_30fps"
NUM_EPISODES = 5
```

- `env_cfg.decimation = 3` → 33Hz physics, 그 사이 한 번씩 렌더 (`render_interval=1`)
- spawn 범위 확장: x, y `±0.15 m`
- **주의**: subtask 관련 observation은 자동 비활성 (없는 환경에서도 안전)

### 5.2 5가지 다양한 spawn 시드

| ep | x_offset | y_offset | yaw |
|----|----------|----------|-----|
| 0 | 0.00 | 0.00 | 0° (정면) |
| 1 | +0.08 | -0.05 | 45° (오른쪽 앞) |
| 2 | -0.08 | +0.05 | -60° (왼쪽 뒤) |
| 3 | +0.05 | +0.08 | 90° (앞-오른쪽 직각) |
| 4 | -0.05 | -0.08 | 135° (뒤-왼쪽) |

→ 정책의 일반화 범위를 한눈에 검증합니다.

### 5.3 SUCCESS / FAIL 판정

각 step마다 다음을 체크:

```
holding = (cube z ≥ 0.15 m) AND (_is_grasped > 0.5) AND (pan_err < 0.262 rad ≈ 15°)
```

`hold_first` (처음 holding 진입한 step) 부터 끝까지의 비율이 **80% 이상** 이면 SUCCESS:

```
ratio = hold_steps / max(MAX_STEPS - hold_first, 1)
success = (hold_first ≥ 0) AND (hold_steps ≥ 5) AND (remaining > 5) AND (ratio ≥ 0.80)
```

### 5.4 저장 포맷

```
/data/infer_v41_30fps/
├── SUCCESS_ep00_x0y0yaw0_ret842/
│   ├── trajectory.npz
│   │   ├─ qpos          (T, 6)   joint position
│   │   ├─ actions       (T, 6)   policy output
│   │   ├─ target_qpos   (T, 6)   delta-action wrapper output
│   │   ├─ cube_pos      (T, 3)   world frame, env origin 보정
│   │   ├─ cube_quat     (T, 4)
│   │   ├─ ep_length     scalar
│   │   └─ total_reward  scalar
│   ├── cam_front.npz    (T, 480, 640, 3) uint8
│   ├── cam_side.npz     (T, 480, 640, 3) uint8
│   ├── cam_wrist.npz    (T, 480, 640, 3) uint8
│   ├── front.mp4        30fps H.264
│   ├── side.mp4
│   └── wrist.mp4
├── FAIL_ep01_...
└── ...
```

### 5.5 실행

```bash
cd /workspace/delivery
export PYTHONPATH=$(pwd):$PYTHONPATH
export DISPLAY=:1
export OMNI_KIT_ACCEPT_EULA=YES

# infer_sac.py 상단의 CKPT/OUTPUT/NUM_EPISODES 수정 후
/isaac-sim/python.sh packages/sim/infer_sac.py
```

진행 로그 예시:

```
[Infer] obs=24, act=6
[Infer] Loaded: /data/rl_sac_state_v41/best_ckpt.pt
  Cube spawn: x=0.000 y=0.000 yaw=0°
  ep0 step=0   z=0.030 pan=0.012 gr=0 hold=0 r=0.0
  ep0 step=20  z=0.045 pan=0.034 gr=0 hold=0 r=18.4
  ep0 step=180 z=0.183 pan=0.121 gr=1 hold=120 r=684.2
  [SUCCESS] ep0 ret=842 ratio=92% hold=140/152
  ...
[Done] 5 episodes, 4 SUCCESS (80%) saved to /data/infer_v41_30fps
```

### 5.6 실제 추론 결과 (`v41_30fps`)

[`inference_results/v41_30fps/`](inference_results/v41_30fps/) 에 v41 best checkpoint의 30fps 추론 결과 5개 SUCCESS 에피소드를 포함했습니다. 각 에피소드 디렉토리에는 `front.mp4`, `side.mp4`, `wrist.mp4` (30fps H.264) 와 `trajectory.npz` (qpos / actions / target_qpos / cube_pose 시계열) 가 들어있고, 카메라 raw `cam_*.npz` (~ 80 MB / camera) 는 repo 용량을 위해 제외했습니다 — 필요하면 동일한 ckpt + `infer_sac.py`로 재생성 가능합니다.

| 에피소드 | spawn (x, y, yaw) | total reward | front | side | wrist |
|---------|-------------------|--------------|-------|------|-------|
| `SUCCESS_ep00_x0y-0yaw161_ret86`  | ≈(0, 0, 161°)  | 86 | [front.mp4](inference_results/v41_30fps/SUCCESS_ep00_x0y-0yaw161_ret86/front.mp4) | [side.mp4](inference_results/v41_30fps/SUCCESS_ep00_x0y-0yaw161_ret86/side.mp4) | [wrist.mp4](inference_results/v41_30fps/SUCCESS_ep00_x0y-0yaw161_ret86/wrist.mp4) |
| `SUCCESS_ep01_x0y-1yaw66_ret67`   | ≈(0, -1cm, 66°) | 67 | [front.mp4](inference_results/v41_30fps/SUCCESS_ep01_x0y-1yaw66_ret67/front.mp4) | [side.mp4](inference_results/v41_30fps/SUCCESS_ep01_x0y-1yaw66_ret67/side.mp4) | [wrist.mp4](inference_results/v41_30fps/SUCCESS_ep01_x0y-1yaw66_ret67/wrist.mp4) |
| `SUCCESS_ep02_x0y-0yaw32_ret62`   | ≈(0, 0, 32°)   | 62 | [front.mp4](inference_results/v41_30fps/SUCCESS_ep02_x0y-0yaw32_ret62/front.mp4) | [side.mp4](inference_results/v41_30fps/SUCCESS_ep02_x0y-0yaw32_ret62/side.mp4) | [wrist.mp4](inference_results/v41_30fps/SUCCESS_ep02_x0y-0yaw32_ret62/wrist.mp4) |
| `SUCCESS_ep03_x0y-0yaw82_ret61`   | ≈(0, 0, 82°)   | 61 | [front.mp4](inference_results/v41_30fps/SUCCESS_ep03_x0y-0yaw82_ret61/front.mp4) | [side.mp4](inference_results/v41_30fps/SUCCESS_ep03_x0y-0yaw82_ret61/side.mp4) | [wrist.mp4](inference_results/v41_30fps/SUCCESS_ep03_x0y-0yaw82_ret61/wrist.mp4) |
| `SUCCESS_ep04_x0y-0yaw-12_ret61`  | ≈(0, 0, -12°)  | 61 | [front.mp4](inference_results/v41_30fps/SUCCESS_ep04_x0y-0yaw-12_ret61/front.mp4) | [side.mp4](inference_results/v41_30fps/SUCCESS_ep04_x0y-0yaw-12_ret61/side.mp4) | [wrist.mp4](inference_results/v41_30fps/SUCCESS_ep04_x0y-0yaw-12_ret61/wrist.mp4) |

> 5/5 SUCCESS (100% hold-ratio ≥ 80%). 각 영상 약 10초 분량(decimation=3 → ~33 Hz, 333 step → 30 fps mp4로 인코딩).
> GitHub UI에서 `.mp4` 링크를 클릭하면 브라우저 내장 플레이어로 바로 재생됩니다.

---

## 6. 학습 → SmolVLA 데이터 변환

### 6.1 PPO/`replay_episodes.py` 산출물 (`/data/rl_output/replay/episode_NNNN/`)

PPO 학습 중 success로 분류된 에피소드는 `episode_tracker.py` 가 분류한 카테고리(`grasp`, `lift3cm`, ..., `lift9cm`, `grasp_hold`)로 `episodes/` 에 저장되고, 그 중 일부가 `replay_episodes.py` 로 mp4까지 만들어진 `replay/` 에 들어갑니다.

```
/data/rl_output/
├── episodes/{grasp,lift3cm,...,lift9cm,grasp_hold}_ep_NNNNNN/
│   ├── frame_NNNN.npz   (qpos, actions, cube_pose + 카메라)
│   └── meta.json
├── replay/episode_NNNN/  (best 20 only, 위와 같은 포맷 + mp4)
└── success_episodes/     (legacy, 사용 안 함)
```

### 6.2 SAC/`infer_sac.py` 산출물 (`/data/infer_v41_30fps/`)

5절 참조. 정확히 30fps이므로 변환 시 `--fps 30` 으로 그대로 사용.

### 6.3 LeRobot v2 변환

```bash
# smolvla 컨테이너에서 (lerobot 이미 설치돼 있어야 함)
python /workspace/vla/convert_replay_to_lerobot.py \
  --replay-dir /data/rl_output/replay \
  --output-dir /data/lerobot_dataset/sim_lift \
  --repo-id  local/sim_lift \
  --fps      30 \
  --cameras  front side          # wrist 도 추가 가능
  --image-size 256 256 \
  --min-reward 0.0
```

이 출력이 SmolVLA 학습 repo의 `data/lerobot_dataset/...` 경로로 들어가 파인튜닝의 입력이 됩니다.

---

## 7. 핵심 변경사항 (leisaac 기본 대비)

자세한 표는 [`delivery/README.md`](delivery/README.md) 의 "핵심 변경사항" 절. 요약:

1. **EE 기준점**: gripper(idx 0) → **jaw(idx 1)** + offset (-0.021, -0.070, 0.02)
2. **모터 스펙**: effort 10→**3.5 Nm**, velocity 10→**6.3 rad/s** (STS3215 실제 스펙)
3. **out_of_reach 기준**: EE↔cube 거리 → **cube 이동량** 기반으로 변경
4. **추가 penalty**: `velocity_excess`, `lateral_deviation`
5. **NaN 방지**: PPO policy std clamp `[0.01, 10]`

---

## 8. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `Xvfb not found` | `apt-get install -y xvfb` (setup_rl_env.sh 가 자동 처리) |
| `EULA not accepted` | `export OMNI_KIT_ACCEPT_EULA=YES` `export ACCEPT_EULA=Y` |
| `--gpus all` 안 잡힘 | `nvidia-container-toolkit` 설치, docker-compose `runtime: nvidia` |
| PPO `normal expects std >= 0` | std clamp 적용돼 있음. 그래도 발생하면 `reward_config.yaml` penalty weight 축소 |
| SAC OOM | `--num-envs` 2048→1024 |
| 추론 스폰이 좁다 | `infer_sac.py` 내 `pose_range["x"]/["y"]` 확장 |
| MP4가 안 나온다 | `cv2.VideoWriter_fourcc(*'mp4v')` 코덱 미설치 — `ffmpeg` 또는 `opencv-python` 재설치 |
| 추론 결과가 항상 FAIL | hold criteria가 너무 엄격할 수 있음. `cube_z ≥ 0.15`, `pan_err < 0.262`, `ratio ≥ 0.80` 완화 |

---

## 9. 라이선스 / 크레딧

- 본 repo의 코드: 사내 사용 목적 (라이선스 별도 협의)
- 의존 외부 프로젝트:
  - [NVIDIA Isaac Sim 5.1](https://developer.nvidia.com/isaac-sim) (Omniverse EULA)
  - [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
  - [leisaac](https://github.com/LightwheelAI/leisaac) (LightwheelAI)
  - [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (ETH Zürich Legged Robotics)
  - [Squint](https://github.com/aalmuzairee/squint) (SAC fast 구현 베이스)
  - [LeRobot](https://github.com/huggingface/lerobot) / [SmolVLA](https://huggingface.co/lerobot/smolvla_base)
