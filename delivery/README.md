# SO101 Lift Cube RL Training

Isaac Lab + leisaac 기반 강화학습으로 SO101 로봇팔이 치약통(cube)을 집어 올리는 정책을 학습합니다.
학습된 데이터는 이후 SmolVLA 학습에 사용됩니다.

---

## 디렉토리 구조

```
delivery/
├── packages/
│   ├── sim/
│   │   ├── train_rl.py          # 메인 학습 스크립트 (PPO + 대시보드 + 데이터 수집)
│   │   ├── env_setup.py         # 환경 설정 (reward/termination 오버라이드, 커리큘럼)
│   │   ├── dashboard.py         # 실시간 웹 대시보드 (카메라 스트리밍 + 학습 현황)
│   │   ├── rewards.py           # 커스텀 reward 함수 (참고용, 실제는 leisaac 내부 사용)
│   │   └── terminations.py      # 커스텀 termination 함수 (참고용)
│   └── common/
│       ├── constants.py         # SO101 관절 이름, 모터 제한값, STS3215 스펙
│       └── robot_utils.py       # leisaac ↔ lerobot 액션 변환 유틸
├── configs/
│   ├── reward_config.yaml       # 전체 학습 설정 (보상, 벌점, 커리큘럼, PPO)
│   └── robot_config.yaml        # SO101 로봇 스펙 참조 문서
├── leisaac_overrides/
│   ├── lerobot.py               # SO101 모터 스펙 수정 (effort 3.5Nm, velocity 6.3 rad/s)
│   └── mdp/
│       ├── rewards.py           # reward 함수 수정본 (jaw 기준 + 커스텀 penalty 추가)
│       └── terminations.py      # termination 함수 수정본 (jaw 기준 + cube 이동 기반)
└── README.md
```

---

## 사전 요구사항

컨테이너에 다음이 설치되어 있어야 합니다:
- Isaac Sim 4.5+
- Isaac Lab 2.3+
- leisaac 0.3.0+
- rsl_rl (PPO)
- PyTorch (CUDA)
- Python 3.11

---

## 설치 (컨테이너 안에서)

### Step 1: 프로젝트 코드 복사

```bash
# delivery.tar.gz를 컨테이너에 복사 후
tar xzf delivery.tar.gz
cd delivery
```

### Step 2: leisaac 파일 덮어쓰기

leisaac 패키지 내 3개 파일을 수정본으로 교체합니다.
이 수정은 아래 변경사항을 반영합니다:
- **모터 스펙**: effort 10Nm → 3.5Nm, velocity 10 → 6.3 rad/s (실제 STS3215 스펙)
- **EE 기준점**: gripper(index 0) → jaw(index 1, 집게 안쪽)
- **커스텀 penalty 함수 추가**: lateral_deviation, velocity_excess, object_out_of_reach(cube 이동 기반)

```bash
# leisaac 설치 경로 확인
LEISAAC_PATH=$(python -c "import leisaac; print(leisaac.__path__[0])")
echo "leisaac path: $LEISAAC_PATH"

# 모터 스펙 수정 (3.5Nm, 6.3 rad/s)
cp leisaac_overrides/lerobot.py $LEISAAC_PATH/assets/robots/lerobot.py

# Reward 함수 수정 (jaw 기준 + 커스텀 penalty)
cp leisaac_overrides/mdp/rewards.py $LEISAAC_PATH/tasks/lift_cube/mdp/rewards.py

# Termination 함수 수정 (jaw 기준 + cube 이동 기반 out_of_reach)
cp leisaac_overrides/mdp/terminations.py $LEISAAC_PATH/tasks/lift_cube/mdp/terminations.py
```

### Step 3: 의존성 설치

```bash
pip install pyyaml  # config 파일 읽기용
```

### Step 4: Headless Display 설정

Isaac Sim은 렌더링을 위해 X display가 필요합니다.

```bash
apt-get update && apt-get install -y xvfb
Xvfb :1 -screen 0 1024x768x24 &>/dev/null &
export DISPLAY=:1
```

---

## 실행

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export DISPLAY=:1
export OMNI_KIT_ACCEPT_EULA=YES

python packages/sim/train_rl.py \
  --config configs/reward_config.yaml \
  --save-dir /data/rl_output
```

### 대시보드

학습이 시작되면 웹 브라우저에서 실시간 모니터링 가능:

```
http://<서버IP>:8888
```

대시보드에서 볼 수 있는 것:
- 카메라 3개 실시간 스트리밍 (front, side, wrist)
- Iteration / Episode / Reward 현황
- Reward History 차트 (20-ep 이동평균 + 최고값)
- 현재 Phase (reaching → grasping → lifting → holding)
- Success Rate
- 커리큘럼 단계 + 현재 cube mass

### 주요 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--config` | `configs/reward_config.yaml` | 학습 설정 파일 |
| `--save-dir` | `/data/rl_output` | 체크포인트 + 성공 데이터 저장 경로 |
| `--resume` | None | 체크포인트에서 이어서 학습 |
| `--num-envs` | (config에서 읽음) | CLI에서 오버라이드 가능 |
| `--http-port` | 8888 | 대시보드 포트 |
| `--max-iterations` | 10,000,000 | 최대 iteration 수 |

---

## 학습 설정 (reward_config.yaml)

모든 하드코딩된 값이 YAML config로 분리되어 있습니다.
학습을 조정하려면 이 파일만 수정하면 됩니다.

### 보상 체계

**긍정 보상 (잘하면 +)** — leisaac 기본값 사용

| # | 이름 | 설명 | 가중치 |
|---|------|------|--------|
| 1 | reach_coarse | jaw가 cube 20cm 이내 접근 | +2 |
| 2 | reach_fine | jaw가 cube 5cm 이내 정밀 접근 | +3 |
| 3 | gripper_open | 물체 근처(10cm) 그리퍼 벌리기 | +3 |
| 4 | gripper_align | 그리퍼 각도(pitch+yaw) 맞추기 | +5 |
| 5 | gripper_close | 물체 가까이(5cm) 그리퍼 닫기 | +10 |
| 6a | grasp_hold | 속도 기반 잡기 감지 | +15 |
| 6b | grasp | 접촉 확인 + 0.5cm 올림 | +25 |
| 7 | lift_low | 5~10cm 들기 (연속) | +20 |
| 8 | lift_high | 10~25cm 들기 (연속) | +20 |
| 9 | hold | 10cm 이상 유지 | +15 |

**벌점 (못하면 -)**

| # | 이름 | 설명 | 가중치 | 에피소드 종료 |
|---|------|------|--------|-------------|
| 10 | drop | 들었다가 떨어뜨림 | -10 | O |
| 11 | out_of_reach | cube가 초기 위치에서 35cm+ 이동 | -5 | O |
| 12 | table_collision | jaw가 테이블(z<5cm)에 박힘 | -5 | O |
| 13 | velocity_excess | 관절 속도 >2 rad/s 초과량 비례 | -0.2 | X |
| 14 | lateral_deviation | gripper가 cube ±20cm 복도 밖 (초기 20k iter만) | -5 | X |

### 커리큘럼

**Mass (물체 무게)**

초기에 무겁게 시작해서 로봇이 밀지 못하게 하고, 점차 가볍게 합니다.

| Until iter | Mass | 설명 |
|-----------|------|------|
| 2,000 | 30kg | 고정 (reach/grip 연습) |
| 5,000 | 15kg | |
| 10,000 | 5kg | |
| 20,000 | 2kg | lift 시작 가능 |
| 50,000 | 1kg | |
| 100,000 | 0.5kg | |
| 이후 | 0.2kg | 실제 치약통 무게 |

**Position & Angle (물체 위치/각도 랜덤화)**

| Until iter | 위치 | 각도 |
|-----------|------|------|
| 5,000 | ±2cm | ±10° |
| 20,000 | ±5cm | ±30° |
| 100,000 | ±10cm | ±60° |
| 500,000 | ±15cm | ±120° |
| 이후 | ±20cm | ±180° |

**Lateral Deviation Penalty**
- 0 ~ 20,000 iter: 활성 (gripper가 cube 주변 ±20cm 복도 밖에 나가면 -5)
- 20,000+ iter: 자동 비활성 (자유 탐색)

### PPO 설정

| 파라미터 | 값 |
|---------|-----|
| learning_rate | 3e-4 (adaptive) |
| gamma | 0.99 |
| lambda | 0.95 |
| clip_param | 0.2 |
| entropy_coef | 0.01 |
| max_grad_norm | 1.0 |
| num_steps_per_env | 48 |
| actor network | [256, 128, 64] ELU |
| critic network | [256, 128, 64] ELU |

---

## 핵심 변경사항 (leisaac 기본 대비)

### 1. EE 기준점 변경
- **기본**: gripper 프레임 (index 0) — wrist 근처
- **수정**: jaw 프레임 (index 1) + offset (-0.021, -0.070, 0.02) — 집게 안쪽
- **이유**: 실제 잡는 지점 기준이어야 reach/grasp 학습이 정확함

### 2. 모터 스펙 현실화
- **기본**: effort 10Nm, velocity 10 rad/s
- **수정**: effort 3.5Nm, velocity 6.3 rad/s (STS3215 실제 스펙)
- **이유**: sim-to-real 전이를 위해 현실과 동일한 물리 적용

### 3. out_of_reach 기준 변경
- **기본**: EE ↔ cube 거리 > 35cm → 종료 (로봇이 멀어져도 죽음)
- **수정**: cube가 초기 위치에서 35cm+ 이동 → 종료 (물체가 밀려난 경우만)
- **이유**: 로봇이 자유롭게 탐색할 수 있어야 reach를 학습함

### 4. 추가된 penalty
- **velocity_excess**: 관절 속도 >2 rad/s 초과분 비례 벌점 (부드러운 동작 유도)
- **lateral_deviation**: gripper가 cube ±20cm 복도 밖 벌점 (초기 20k iter, 방향 학습 가이드)

### 5. NaN crash 방지
- PPO policy의 std를 [0.01, 10]으로 clamp
- 큰 벌점으로 인한 gradient 폭발 → std가 NaN 되는 문제 해결

---

## 성공 기준 (데이터 저장)

다음 조건을 **연속 10 스텝** 유지하면 성공 에피소드로 저장:
- cube 높이 > 10cm
- gripper 닫힘 (pos < 0.26)
- EE-cube 거리 < 5cm
- contact force > 0.1N

성공 데이터는 `--save-dir` 경로에 저장됩니다.

---

## 체크포인트

- 50 iteration마다 자동 저장: `{save-dir}/logs/model_{iter}.pt`
- 최신 모델: `{save-dir}/logs/model_latest.pt`
- 이어서 학습: `--resume {save-dir}/logs/model_latest.pt`

---

## 속도 최적화 설정

| 항목 | 기본값 | 현재 설정 | 효과 |
|------|--------|----------|------|
| num_envs | 1 | 3 | 3배 병렬 |
| render_interval | 8 | 16 | 렌더링 절반 |
| camera resolution | 640x480 | 320x240 | 픽셀 1/4 |

카메라는 학습(observation은 joint pos)에 안 쓰이고 대시보드 + 데이터 수집용이므로,
해상도를 낮춰도 학습 품질에 영향 없음.

---

## 트러블슈팅

### Xvfb가 없으면
```bash
apt-get install -y xvfb
```

### EULA 에러
```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
```

### GPU 안 잡힘
```bash
# 컨테이너 실행 시 --gpus all 필요
docker run --gpus all ...
```

### NaN crash (normal expects std >= 0)
이미 코드에 std clamp가 적용되어 있어 발생하지 않아야 합니다.
만약 발생하면 reward_config.yaml에서 penalty weight를 줄여보세요.
