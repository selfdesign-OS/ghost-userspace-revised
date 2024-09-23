#!/bin/bash

# 라운드로빈 방식으로 실행될 프로세스 실행
for i in {1..5}; do
    ./test_program &  # test_program은 위의 C 코드로 컴파일한 실행 파일
done