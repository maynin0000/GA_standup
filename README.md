# 유전 알고리즘으로 일어서는 로봇 학습하기

PyBullet 물리 시뮬레이터에서 단순한 인체형 로봇을 만들고, 유전 알고리즘으로 일어서는 동작을 탐색하는 예제입니다.

처음 버전처럼 관절 토크를 매 순간 자유롭게 던지면 움직임이 너무 비인간적으로 튀기 쉽습니다. 그래서 현재 코드는 사람 동작에 더 가까운 6개의 단계로 움직임을 쪼갭니다.

유전 알고리즘은 앞의 4개 동작 순서와 각 동작 강도를 학습합니다. 마지막 2개 동작인 `extend_legs`, `stabilize`는 항상 끝에 고정됩니다.

모형은 골반, 상체, 머리, 양팔, 양다리, 발 링크를 가진 간단한 humanoid 구조입니다. 학습 평가는 한 개체씩 따로 돌리지 않고, 한 세대의 population을 같은 물리 월드에 나란히 배치해서 동시에 시뮬레이션합니다.

## 설치

현재 프로젝트의 `.venv`는 Python 3.10 기반입니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

가상환경을 새로 만든다면:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyBullet이 wheel로 설치되지 않으면 Visual Studio C++ Build Tools가 필요할 수 있습니다.

## 실행

학습만 빠르게 돌립니다.

```powershell
.\.venv\Scripts\python.exe .\ga_standup.py --generations 30 --population 24
```

population 전체가 동시에 평가되는 모습을 보고 싶다면:

```powershell
.\.venv\Scripts\python.exe .\ga_standup.py --generations 3 --population 8 --watch-population
```

학습 후 PyBullet GUI로 최고 개체를 반복 재생합니다.

```powershell
.\.venv\Scripts\python.exe .\ga_standup.py --generations 30 --population 24 --replay
```

빠르게 확인하려면:

```powershell
.\.venv\Scripts\python.exe .\ga_standup.py --generations 2 --population 4 --steps 120 --replay
```

## 6개 동작 단계

현재 유전자는 임의 토크 배열이 아니라 아래 6개 동작의 순서와 강도 값입니다.

1. `sit_up`: 누운 상태에서 상체를 일으킵니다.
2. `fold_right_leg`: 오른쪽 다리를 먼저 접어 몸쪽으로 가져옵니다.
3. `fold_left_leg`: 왼쪽 다리도 접어 발을 디딜 준비를 합니다.
4. `plant_feet`: 두 발을 바닥에 디디는 자세를 만듭니다.
5. `extend_legs`: 무릎과 고관절을 펴며 몸을 세웁니다.
6. `stabilize`: 선 자세에 가깝게 안정화합니다.

GUI 재생 중에는 현재 단계 이름이 화면에 표시됩니다.

학습 중에는 각 세대의 최고 개체 순서가 터미널에 출력됩니다.

```text
order=fold_left_leg>sit_up>fold_right_leg>plant_feet>extend_legs>stabilize
```

## 선택 방식

점수는 머리의 평균 높이를 중심으로 계산합니다. 한 세대 평가가 끝나면 상위 2개 유전자만 부모 풀로 사용하고, 다음 세대는 이 둘의 교차와 변이로 다시 생성합니다.

## 코드에서 조정할 부분

- `MOVEMENT_PRIMITIVES`: 6개 동작의 목표 관절 각도와 지속 시간을 정의합니다.
- `Genome`: 동작 순서와 각 동작 단계의 강도를 담습니다.
- `create_humanoid_robot()`: 골반, 상체, 머리, 팔, 다리, 발 링크를 만듭니다.
- `evaluate_population()`: population 전체를 같은 월드에서 동시에 평가합니다.
- `fitness()`: 머리 평균 높이를 중심으로 점수를 계산합니다.
- `mutate()`, `crossover()`: 유전 알고리즘의 변이와 교차 방식입니다.
- `JOINT_FORCE_LIMIT`, `DEFAULT_STEPS`: 관절 힘과 한 번 평가할 시뮬레이션 길이입니다.

## 다음 개선 아이디어

- 팔 링크를 추가해서 바닥을 짚고 일어나게 만들기
- 어깨/고관절에 회전축을 추가해서 팔 짚기와 다리 꼬기를 더 자연스럽게 만들기
- 발 접촉 여부를 보상 함수에 더 강하게 반영하기
- 각 단계의 지속 시간도 유전자가 학습하게 만들기
