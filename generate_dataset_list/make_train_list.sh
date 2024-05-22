#!/bin/bash

# 설정할 디렉토리 경로
audio_dir="/mnt/f/fluent/037.교육용_한국인의_영어_음성_데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_PRN_en_NA_NA"
label_dir="/mnt/f/fluent/037.교육용_한국인의_영어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_PRN_en_NA_NA"
output_dir="."  # 파일을 저장할 경로로 변경

# 출력 파일 초기화
> "$output_dir/pron_trn.list"
> "$output_dir/pron_val.list"
> "$output_dir/pron_test.list"

# 트레인, 밸리데이션, 테스트 파일 리스트
train_list="$output_dir/pron_trn.list"
val_list="$output_dir/pron_val.list"
test_list="$output_dir/pron_test.list"

# 모든 .wav 파일에 대해 처리
find "$audio_dir" -name "*.wav" | while read -r wav_file; do
    # 대응하는 json 파일 경로 설정
    filename=$(basename "$wav_file" .wav)
    json_file="$label_dir/$filename.json"
    
    # json 파일이 존재하는지 확인
    if [ -f "$json_file" ]; then
        # 점수와 스크립트 추출
        articulation_score=$(jq -r '.itemScript.rating.articulationScore' "$json_file")
        prosody_score=$(jq -r '.itemScript.rating.prosodyScore' "$json_file")
        script=$(jq -r '.itemScript.contents[0].sentence' "$json_file")
        
        # 결과 문자열 생성
        result="$wav_file\t$articulation_score\t$prosody_score\t$script"
        
        # 임의로 파일을 트레인, 밸리데이션, 테스트 셋에 분류
        hash_value=$(echo -n "$filename" | md5sum | cut -d' ' -f1)
        mod_value=$((0x${hash_value:0:2} % 10))
        
        if [ "$mod_value" -lt 7 ]; then
            echo -e "$result" >> "$train_list"
        elif [ "$mod_value" -lt 9 ]; then
            echo -e "$result" >> "$val_list"
        else
            echo -e "$result" >> "$test_list"
        fi
    else
        echo "Warning: $json_file not found."
    fi
done

echo "Files generated: pron_trn.list, pron_val.list, pron_test.list"
