
#! /bin/bash

if [ $# -ne 1 ]; then
	echo "rule : ./matrix.sh (filename)"
	exit 1
fi

while read line
do
	fst=$(echo ${line%%'$'*})
	mid=$(echo ${line%'$'*})
	mid=$(echo ${mid#*'$'})
	lst=$(echo ${line##*'$'})
	
	echo "$fst $mid $lst $row"
	
	column1+=($fst)
	column2+=($mid)
	column3+=($lst)
	
done < "$1"

column=(column1 column2 column3)

for i in "${column[@]}"
do
	col="$i[@]"
	sum=0
	for j in ${!col}
	do
		sum=$(($sum + $j))
	done
	sum_arr+="$sum "
done

echo ${sum_arr[0]}
