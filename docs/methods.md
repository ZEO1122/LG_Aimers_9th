# 설계: 정보의 시점과 평가 행 독립성

각 투구의 성공 확률을 예측할 때는 투구 직전의 정보만 입력으로 사용할 수 있어야 합니다. 이 프로젝트에서는 피처의 유용성뿐 아니라 **언제 알 수 있는 값인지**, **다른 평가 행에 의존하는지**를 함께 검토했습니다.

## 공개 구현

[공통 파이프라인](../Preprocess/common/pipeline.py)은 `fit(train_rows, history_rows)`에서 상태를 만들고 `transform(rows)`에서 이를 조회합니다. 학습 데이터와 이력을 DataFrame으로 받으며, 모델 학습이나 예측 확률 결합은 수행하지 않습니다.

| 모듈 | 역할 | 정보 경계 |
|---|---|---|
| [base.py](../Preprocess/common/base.py) | 경기 상황과 공식 `asof_*` 값, 선수별 과거 상태 변환 | 예측 연도보다 앞선 시즌의 snapshot 조회 |
| [history.py](../Preprocess/common/history.py) | 익명 ID 대응과 TrackMan 이력 집계 | 예측 연도 이전 이력으로 조회표 구성 |
| [domain.py](../Preprocess/common/domain.py) | 카운트·주자·손잡이 등 상황 파생값 | 현재 행의 투구 이전 입력 사용 |
| [state.py](../Preprocess/common/state.py) | 투수 상태와 카운트의 상호작용 | 변환된 현재 행 피처 사용 |
| [mechanics.py](../Preprocess/common/mechanics.py) | 과거 구속·릴리스·무브먼트 특성 구간화 | 이전 이력과 학습 시 고정한 경계 사용 |
| [tensor.py](../Preprocess/common/tensor.py) | 수치 표준화와 범주 인코딩 | 학습 평균·표준편차·범주 사전 고정 |

`transform`은 타깃 열을 제거합니다. 평가 배치에서 새 빈도·평균·누적 이력을 계산하지 않고, 고정된 상태와 각 행의 입력값을 사용합니다. 알 수 없는 범주는 학습 시 정한 대체 토큰으로 처리합니다.

이력 조회에 과거 시점 조건이 있더라도 검증 fold 전체를 대상으로 `fit`하면 안 됩니다. 수치 표준화·범주 사전 등도 검증 데이터의 영향을 받지 않도록 **fold마다 새 파이프라인을 만들고 학습 행에만 적합**해야 합니다.

## 합성 입력으로 확인하는 계약

[테스트](../Preprocess/tests/test_preprocessing_contract.py)는 실제 대회 행을 담지 않고 작은 DataFrame을 직접 생성합니다.

| 검사 | 확인하는 성질 |
|---|---|
| 배치와 단일 행 비교 | 동일한 행의 표 형식 피처와 텐서가 일치 |
| 행 순서 변경 | 입력을 섞어도 원래 행에 대응하는 피처가 일치 |
| 타깃 열 포함·제거 | 평가 입력의 타깃 열이 기본 피처를 바꾸지 않음 |
| 현재·미래 시즌 이력 변조 | 이후 시즌 값을 바꿔도 해당 cutoff의 mechanics 조회표가 일치 |

배치·행 순서 테스트는 `use_trackman=False` 설정이며, TrackMan 관련 검사는 별도 합성 mechanics 이력으로 수행합니다. 전체 TrackMan ID 대응 파이프라인이나 모든 누수 가능성을 검증한 것으로 확대 해석하지 않습니다.

## 시간 순서 검증

원본 모델 실험에서는 검증 연도보다 이전의 연도로 학습하는 방식을 사용했습니다. 대표 기준 실험 ML-015의 구성은 다음과 같습니다.

| 학습 연도 | 검증 연도 |
|---|---|
| 2019–2021 | 2022 |
| 2019–2022 | 2023 |
| 2019–2023 | 2024 |

연도별 Brier와 경기 유형별 결과를 함께 확인하고, 여러 seed의 예측 평균을 비교했습니다. 후속 실험에서 이미 여러 번 관찰한 2024 결과는 새로운 미관측 검증이 아닌 **사후 감사**로 기록했습니다. 수치와 해석은 [대표 실험](experiments.md)에 있습니다.

## EDA와 재현성 기록

원본 EDA에서는 결측치·자료형·타깃 및 시즌 분포·경기 상황·선수 이력·TrackMan 결합 가능성을 점검했습니다. 원본 노트북 `EDA/comprehensive_eda.ipynb`는 비공개 연구 저장소에 그대로 보존하며, 이 공개본에는 포함하지 않습니다.

연구 과정에서는 데이터·피처 순서·소스의 해시, seed, 학습 설정을 고정해 비교 기준을 만들고, 제출 후보에서는 원래 계산과 패키징된 계산의 일치 여부를 별도로 검사했습니다. 해당 전체 실험 실행기와 산출물은 이 공개본에 포함되지 않습니다.

원본 기록 출처: `Preprocess/README.md`, `Experiment/ML/EXP_ML_015_reproducible_baseline_contract/README.md`, `Experiment/Ensemble/Existing_ENS/EXP_ENS_045_ens044_production_runtime/README.md`.
