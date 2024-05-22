# pronunciation_assessment

### Language

    영어, 중국어, 일본어, 독일어, 프랑스어, 스페인어, 러시아어


### Docker image 생성
```
docker build -t nia_pron .
```

### Docker container 생성
```
docker run -it -v /data1/nia_db:/data/project/nia/pron/data --gpus all --shm-size 10gb nia_pron
```
-v/data1/nia_db:/data/project/nia/pron/data : 다운받은 데이터가  포함된  “/data1/nia_db” 폴더를 docker container 안의 “/data/project/nia/pron/data” 폴더와 공유



## Train
```
python train_mlp.py --lang='de' --label_type1='pron' --label_type2='articulation' --dir_data='data/audio' --dir_list='data_prepare' --mlp_hidden=64 --epochs=200 --patience=20 --batch_size=256 --dir_model='model'
```
--lang : 언어 선택 (de-독일어, en-영어, es-스페인어, fr-프랑스어, jp-일어, ru-러시아어, zh-중국어)

--label_type1 : 학습할 발음평가 라벨 선택(1) ('pron' / 'fluency')

--label_type2 : 학습할 발음평가 라벨 선택(2) ('articulation' / 'prosody')

--dir_data : 학습할 발음평가 데이터가 위치한 폴더 경로

--dir_list : 과정 3에서 준비한 trn/val/test list가 있는 폴더 경로

--dir_model : inference를 수행할 모델 경로 

--mlp_hidden : 모델의 hidden units 개수

--epochs : finetune 훈련 횟수

--patience : 성능이 더이상 오르지 않으면 학습을 조기 종료하기 위한 파라미터

--batch_size : 훈련 batch 크기

## Inference wav
```
python inference_wav.py --wav=dir_wav --lang='de' --label_type1='pron' --label_type2='articulation' --device='cpu' --dir_model='model'
```
--wav : wav 파일 경로

--lang : 언어 선택 (de-독일어, en-영어, es-스페인어, fr-프랑스어, jp-일어, ru-러시아어, zh-중국어)

--label_type1 : 평가할 발음평가 라벨 선택(1) ('pron' / 'fluency')

--label_type2 : 평가할 발음평가 라벨 선택(2) ('articulation' / 'prosody')

--device : inference를 수행할 device 선택 ('cuda' / 'cpu')

--dir_model : inference를 수행할 모델 경로

## Inference list file
```
python inference.py --lang='de' --label_type1='pron' --label_type2='articulation' --dir_list=data_prepare --device='cuda' --dir_data='data/audio' --data_type='test' --dir_model='model'
```
--lang : 언어 선택 (de-독일어, en-영어, es-스페인어, fr-프랑스어, jp-일어, ru-러시아어, zh-중국어)

--label_type1 : 평가할 발음평가 라벨 선택(1) ('pron' / 'fluency')

--label_type2 : 평가할 발음평가 라벨 선택(2) ('articulation' / 'prosody')

--dir_list : 과정 3에서 준비한 리스트가 위치한 폴더 경로

--device : inference를 수행할 device 선택 ('cuda' / 'cpu')

--dir_data : 학습할 발음평가 데이터가 위치한 폴더 경로

--data_type : 평가할 데이터 타입 (trn / val / test)

--dir_model : inference를 수행할 모델 경로
