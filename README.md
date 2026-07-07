# 유전 알고리즘으로 일어서는 로봇 학습하기

PyBullet 물리 시뮬레이션에서 간단한 humanoid 로봇을 만들고, 유전 알고리즘으로 누운 상태에서 일어서는 동작을 탐색하는 예제입니다.

현재 유전자는 완전한 자유 관절 제어가 아니라, 미리 정의된 6개의 동작 primitive를 조합합니다. 앞의 4개 동작 순서와 각 동작의 강도를 학습하고, 마지막 2개 동작인 `extend_legs`, `stabilize`는 항상 끝에 고정됩니다.

로봇은 공중에서 떨어지며 시작하지 않고, 하늘을 보고 누운 자세로 바닥에 닿은 상태에서 바로 시작합니다. 모델 좌표계는 처음부터 누운 인체 기준으로 구성되어, X축은 머리/발 방향이고 Z축은 위쪽입니다.

## 설치

권장 환경은 Python 3.10 계열 가상환경입니다.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyBullet wheel이 설치되지 않으면 Visual Studio C++ Build Tools가 필요할 수 있습니다.

## 실행

기본값은 population 100, 상위 parent pool 10, elite 5입니다.

```powershell
.\.venv\Scripts\python.exe .\ga_standup.py --generations 30
```

population 전체가 동시에 평가되는 모습을 보고 싶다면:

```powershell
.\.venv\Scripts\python.exe .\ga_standup.py --generations 3 --population 20 --watch-population
```

학습 후 최고 개체를 PyBullet GUI로 반복 재생하려면:

```powershell
.\.venv\Scripts\python.exe .\ga_standup.py --generations 30 --replay
```

빠르게 확인하려면:

```powershell
.\.venv\Scripts\python.exe .\ga_standup.py --generations 2 --population 8 --steps 120 --replay
```

## 학습 방식

각 개체는 다음 값을 가집니다.

- `order`: 앞의 4개 primitive 순서
- `weights`: 각 primitive 목표 관절 각도의 강도 배율

다음 세대는 점수 상위 `--parent-pool`개 개체를 부모 후보로 삼고, tournament selection, crossover, mutation으로 생성합니다. 기본 parent pool은 10개입니다.

## 6개 동작 primitive

1. `sit_up`: 누운 상태에서 상체를 세웁니다.
2. `fold_right_leg`: 오른쪽 다리를 먼저 접습니다.
3. `fold_left_leg`: 왼쪽 다리를 접습니다.
4. `plant_feet`: 양발을 바닥 쪽에 두는 자세를 만듭니다.
5. `extend_legs`: 무릎과 고관절을 펴며 몸을 세웁니다.
6. `stabilize`: 관절 목표를 중립에 가깝게 돌려 안정화합니다.

## 평가 점수

점수는 다음 요소를 함께 봅니다.

- 평균 머리 높이
- 마지막 머리 높이
- 골반/몸통의 upright 정도
- 발이 바닥에 닿았는지
- 마지막 속도가 낮아 안정적인지
- 시작 위치에서 너무 멀리 밀려나지 않았는지
- 관절 모터가 사용한 대략적인 에너지

population은 같은 PyBullet 월드에 동시에 놓이지만, 로봇끼리의 충돌은 꺼서 서로 점수에 영향을 주지 않도록 했습니다.

## 조정할 만한 값

- `--population`: 기본 100
- `--parent-pool`: 기본 10
- `--elites`: 기본 5
- `--mutation-rate`: 기본 0.12
- `--mutation-scale`: 기본 0.22
- `--steps`: 기본 480

## 다음 개선 아이디어

- 팔/손 링크를 더 적극적으로 활용해서 바닥을 짚고 일어나게 만들기
- 어깨, 고관절에 더 많은 회전축을 추가해 움직임을 자연스럽게 만들기
- primitive duration도 유전자에 포함하기
- 일정 점수 이상인 genome을 파일로 저장하고 이어서 학습하기
