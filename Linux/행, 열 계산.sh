#!/bin/bash

# 파일에서 데이터 읽기 및 배열 생성
ArrCount=1
sum1=0
sum2=0

while IFS="$" read -r -a line; do
    for ((i=0; i<${#line[@]}; i++)); do
        if [ $ArrCount -eq 1 ]; then
            score1[$i]=${line[$i]}
        else
            score2[$i]=${line[$i]}
        fi
    done

    let ArrCount++
done < "$1"

# 배열 합계 계산
num=${#score1[@]}

# score1의 열 합계 계산
for ((i=0; i<num; i++)); do
    sum1=$(($sum1 + ${score1[$i]}))
done

# score2의 열 합계 계산
for ((i=0; i<num; i++)); do
    sum2=$(($sum2 + ${score2[$i]}))
done

sum4=$((${sum1}+${sum2}))

# 결과 출력
for ((i=0; i<num; i++)); do
    echo -n "${score1[$i]} "
done
echo -n "$sum1"
echo ""

for ((i=0; i<num; i++)); do
    echo -n "${score2[$i]} "
done
echo -n "$sum2"
echo ""

for ((i=0; i<num; i++)); do
    sum3[$i]=$((${score1[$i]} + ${score2[$i]}))
    echo -n "${sum3[$i]} "
done
echo -n "$sum4"
echo ""
