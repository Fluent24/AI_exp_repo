# Fluent AI 실험 저장소

이 저장소는 오디오 데이터를 사용하여 다양한 알고리즘(SVR 및 XGBoost)을 통해 머신러닝 모델을 학습하고 평가하는 코드와 데이터셋을 포함하고 있습니다. 주요 초점은 Wav2Vec2를 사용하여 특징을 추출하고 이를 다양한 회귀 작업에 사용하는 것입니다.

## 목차
- [Fluent AI 실험 저장소](#fluent-ai-실험-저장소)
  - [목차](#목차)
  - [프로젝트 구조](#프로젝트-구조)
  - [설치](#설치)
  - [사용법](#사용법)
    - [특징 추출](#특징-추출)
    - [모델 학습](#모델-학습)
  - [스크립트 설명](#스크립트-설명)
  - [라이센스](#라이센스)

## 프로젝트 구조

```
.
├── datasets_list/lang_en                # 데이터셋 목록 (음성파일의 위치와 점수를 기록)
├── datasets_sample/lang_en              # 샘플 데이터셋 (작은량의 데이터셋으로 실험가능)
├── generate_dataset_list                # 데이터셋 목록 생성 스크립트
├── guide                                # AI_hub의 문서 및 가이드
├── temp                                 # 임시 파일
├── .gitignore                           # Git ignore 파일
├── README.md                            # 프로젝트 README
├── compare_3arch.sh                     # 3가지 아키텍처를 비교하고 파라미터튜닝하는 셸 스크립트
├── compare_models.py                    # 모델을 비교하는 파이썬파일
├── dataset_tree.txt                     # 데이터셋 디렉토리 트리
├── requirements.txt                     # 파이썬 의존성 목록
├── score_model.py                       # 모델을 평가하는 파이썬 스크립트
├── train_SVR_hyper.py                   # 하이퍼파라미터 튜닝을 포함한 SVR 학습 스크립트
└── train_XGB_hyper.py                   # 하이퍼파라미터 튜닝을 포함한 XGBoost 학습 스크립트
```

## 설치

필요한 의존성을 설치하려면 다음 명령을 실행하세요:

```bash
pip install -r requirements.txt
```


## 사용법

### 특징 추출

Wav2Vec2를 사용하여 오디오 파일에서 특징을 추출하려면, `train_XGB_hyper.py` 스크립트 내의 `feat_extraction` 함수를 실행하세요. 데이터셋이 올바르게 구성되고 경로가 스크립트에 맞게 설정되어 있는지 확인하세요.

### 모델 학습

모델을 학습하려면 각 학습 스크립트를 사용하세요:

- **하이퍼파라미터 튜닝을 포함한 SVR**:
  ```bash
  python train_SVR_hyper.py --dir_list="datasets_list" --dir_model="model_svr" --label_type1="pron" --label_type2="prosody"
  ```

- **하이퍼파라미터 튜닝을 포함한 XGBoost**:
  ```bash
  python train_XGB_hyper.py --dir_list="datasets_list" --dir_model="model_xgb" --label_type1="pron" --label_type2="prosody"
  ```


## 스크립트 설명


- 딥러닝
  - `compare_3arch.sh`: 세 가지 다른 아키텍처를 비교하는 셸 스크립트.
  - `score_model.py`: 스코어링 딥러닝 모델 class 정의
  - `compare_models.py`: 다양한 모델을 학습하는 파이썬 파일.
- 머신러닝
  - `train_SVR_hyper.py`: 하이퍼파라미터 튜닝을 포함한 SVR 모델을 학습하는 스크립트.
  - `train_XGB_hyper.py`: 하이퍼파라미터 튜닝을 포함한 XGBoost 모델을 학습하는 스크립트.

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
