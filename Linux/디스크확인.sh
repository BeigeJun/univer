#!/bin/bash
calc()
{ awk 'BEGIN {printf "%.2f\n", '"$*"'; exit}'
}
#set -x # trace on script excution
echo "--------------------------------------------------"
echo " Summary of Disk Usages per User "
echo "--------------------------------------------------"
printf "%-15s %-25s %s\n" Lectuer Directory Used
echo "--------------------------------------------------"
IFS=:
cut -d: -f1,2 /home/seojun/workshop/doit/data/contents.dir > ~/tmppasswd

total=0
while read Lecture Directory Used
do
	dush=`du -sh /home/seojun/workshop/doit/data/$Directory 2>/dev/null | cut -f1 `
	dus=`du -s /home/seojun/workshop/doit/data/$Directory 2>/dev/null | cut -f1 `
	total=$(( $total + $dus ))
	printf "%-15s %-25s %8s\n" $Lecture $Directory $dush
done < ~/tmppasswd

echo "--------------------------------------------------"
totalg=$(($total / 1024 / 1024))
totalm=$(($total / 1024 ))
totalgb=$(calc "$total / (1024 * 1024)")
printf "Total : %10dK %10dM       %4.1fG\n" $total $totalm $totalg
echo "--------------------------------------------------"
rm -f ~/tmppasswd
