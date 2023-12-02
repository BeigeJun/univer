#!/bin/bash

while true; do
    # 사용자로부터 파일 형식 선택 받기
    read -p "파일 형식(mp3, mp4, jpg)'q'로 종료: " file_type
    if [ "$file_type" == "q" ]; then
        echo "종료"
        exit 0
    fi

    read -p "디렉터리: " directory_name
    read -p "재생할 파일: " target_file_name

    target_file="/home/seojun/workshop/w13/$directory_name/$target_file_name.$file_type"

    if [ ! -f "$target_file" ]; then
        echo "파일이 존재하지 않습니다: $target_file"
        continue
    fi

    execute_file() {
        case $file_type in
            "mp3")
                mpg123 "$target_file"
                ;;
            "mp4")
                cvlc "$target_file"
                ;;
            "jpg")
                xdg-open "$target_file" || open "$target_file"
                ;;
            *)
                echo "지원하지 않는 확장자: $file_type"
                ;;
        esac
    }
    execute_file
done
