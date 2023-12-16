#!/bin/bash
total_dir_first=0
total_dir_second=0
dir_count=0
calc() {
  awk "BEGIN {printf \"%.2f\n\", $1; exit}"
}
Used_display() {
  while read Lecture Directory
  do
    dush=$(du -sh "$1/$Directory" 2>/dev/null | cut -f1)
    dus=$(du -s "$1/$Directory" 2>/dev/null | cut -f1)
    dir_count=$(count_files "$1/$Directory")
    total_dir_second=$((dir_count + total_dir_second))
    total=$((dus + total))
    let total_dir_first++
    printf "%-15s %-19s %-9s %s\n" "$Lecture" "$Directory" "$dir_count" "$dush"
    for file in "$1/$Directory"/*
    do
      file_name=${file##*/}
      file_size=$(du -b "$file" 2>/dev/null | cut -f1)
      format_num=$(format_number "$file_size")
      printf "%37s %12s\n" "$file_name" "$format_num"
    done
  done < ~/tmpcontents.dir
}
count_files() {
  find "$1" -type f | wc -l
}
format_number() {
  echo "$1" | awk '{printf "%'\''d\n", $1}'
}
echo "--------------------------------------------------"
echo "            Disk Usages per Lecture"
echo "--------------------------------------------------"
printf "%-15s %-15s %-13s %s\n" Lecture Directory Files Used
echo "--------------------------------------------------"
IFS=:
cut -d: -f1,2 "$1/contents.dir" > ~/tmpcontents.dir
total=0
Used_display "$1"
echo "--------------------------------------------------"
totalg=$((total / 1024 / 1024))
totalm=$((total / 1024))
totalgb=$(calc "$total / (1024 * 1024)")
printf "Total : %9s %19s %11.2fM\n" "$total_dir_first" "$total_dir_second" "$totalm"
echo "--------------------------------------------------"
rm -f ~/tmpcontents.dir
