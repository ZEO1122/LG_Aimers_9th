# 실행과 재현 범위

## 검증 가능한 실행

Python 3.11 환경에서 저장소 루트를 작업 디렉터리로 사용한다.
기존 프로젝트의 전처리 의존성인 NumPy 1.26.4와 pandas 2.0.3만 필요하다.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s Preprocess/tests -v
python -m unittest discover -s tests -v
python tools/check_publication.py
python Preprocess/run.py --help
```

배포 준비 시 동일한 패키지 버전의 기존 Python 3.11 환경에서 합성 테스트를 실행했다.
새 머신의 설치와 테스트는 GitHub Actions에도 구성했다. 로컬 테스트와 원격 CI 결과는 구분한다.

| 검사 | 확인하는 내용 | 확인하지 않는 내용 |
|---|---|---|
| 전처리 합성 테스트 | batch/singleton, 행 순서, 평가 target, mechanics cutoff | 전체 데이터의 모든 경계 사례 |
| 업로드 검사 | 선별 경로, 파일 형식, 대표 인증키 패턴 | 데이터·코드의 공개 권한, 모든 비밀정보 |
| 업로드 검사 회귀 테스트 | 강제로 추적한 CSV, 노트북·모델·심볼릭 링크 차단 | 원본 연구 저장소의 과거 이력 |
| CLI 도움말 | 보존한 실행 진입점의 import·옵션 확인 | 실제 CSV smoke 실행 |

## 전체 대회 점수 재현

이 저장소에는 최종 앙상블, 학습 가중치, 공식 데이터, OOF 예측 및 lookup이 포함되지 않는다.
따라서 clone 후 실행하는 것으로 첨부 리더보드 점수가 재현되지는 않는다.
최종 제출물과 해당 점수의 연결도 아직 확정하지 않았으므로 특정 실험을 최종 모델로 지정하지 않았다.

공식 데이터가 필요한 원본 학습·평가 코드는 기존 Private 연구 저장소에 보존한다.
대회 데이터의 포트폴리오 목적 재학습이나 제3자 제공을 허용한다고 해석하지 않는다.
이 공개 준비본은 데이터가 없는 코드 검토와 합성 입력 검증을 위한 범위다.

연구 결과는 [실험 요약](experiments.md)에 당시 기록임을 표시했다.
이번 정리 과정에서는 공식 데이터 재학습, EDA 재실행 또는 제출 패키지 재추론을 수행하지 않았다.
