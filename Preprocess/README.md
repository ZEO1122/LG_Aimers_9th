# 투구 이전 정보의 공통 전처리

`fit`에서 학습한 상태를 고정하고 `transform`에서 각 평가 행을 독립적으로 변환한다.
원본 연구 저장소의 전처리 Python 코드와 합성 테스트를 변경 없이 선별했다.

## 모듈

| 파일 | 역할 |
|---|---|
| `common/base.py` | 경기 상황과 공식 과거 이력 피처 변환 |
| `common/history.py` | cutoff 이전 이력 집계와 익명 ID 매핑 |
| `common/domain.py` | 경기 상황·투구 의도 피처 |
| `common/state.py` | 상태와 카운트 상호작용 |
| `common/mechanics.py` | 과거 TrackMan 이력의 mechanics bin |
| `common/tensor.py` | 학습 통계·범주 사전 기반 tensor 변환 |
| `common/pipeline.py` | 전처리 단계 연결 |

## 데이터 없이 확인하기

저장소 루트에서 Python 3.11과 `requirements.txt`의 환경을 사용한다.

```bash
python -m unittest discover -s Preprocess/tests -v
```

테스트는 코드에서 직접 만든 8개의 합성 행을 사용한다. 공식 데이터를 읽지 않는다.
batch/singleton 일치, 행 순서 불변, 평가 target 무시, mechanics lookup의 당해·미래 시즌 차단을 검사한다.
합성 테스트 통과는 전체 학습·추론 패키지의 점수 재현을 뜻하지 않는다.

## API

`PreprocessingPipeline.fit(train_rows, history_rows)`에서 학습 상태를 정하고,
`transform(rows)`으로 `base`, `history_union`, `semantic`, `deep_frame`, `intent_frame`,
`state_features`, `deep_tensor`를 생성한다. TrackMan 옵션을 켜면 `mechanics_bins`도 생성한다.
공식 데이터 사용이 허용된 환경에서는 fold마다 새 pipeline을 만들고 과거 학습 행에만 fit한다.

`run.py`는 원본의 CSV smoke 진입점을 보존한 파일이다. 실제 데이터가 필요하므로
포트폴리오의 기본 실행 경로에는 포함하지 않았다. 옵션만 확인하려면 다음을 실행한다.

```bash
python Preprocess/run.py --help
```

입력 데이터와 학습된 lookup·tensor·encoder는 Git에 포함하지 않는다.
공개 범위와 실행 한계는 [재현 안내](../docs/reproduction.md)를 참고한다.
