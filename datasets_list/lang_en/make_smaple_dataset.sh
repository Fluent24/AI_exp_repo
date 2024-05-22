#!/bin/bash

# pron_trn.list 파일에서 첫 7,000줄 읽기
head -n 70 pron_trn.list > pron_trn_70.list
head -n 20 pron_val.list > pron_val_20.list
head -n 10 pron_test.list > pron_test_10.list