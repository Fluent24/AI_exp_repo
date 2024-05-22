#!/bin/bash

# pron_trn.list 파일에서 첫 7,000줄 읽기
head -n 7000 pron_trn.list > pron_trn_7000.list
head -n 2000 pron_val.list > pron_val_2000.list
head -n 1000 pron_test.list > pron_test_1000.list