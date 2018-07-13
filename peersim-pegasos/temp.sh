sum=0
while read p; do
  x=`echo $p | awk '{print $6}'`
 ((sum+=$x))
done < "temp.txt"
echo $sum
