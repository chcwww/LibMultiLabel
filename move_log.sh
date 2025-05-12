dir=$1
mkdir $dir $dir/memoery_dat
mv *.dat $dir/memory_dat/
mv memory_profile $dir/
mv para_log $dir/
mv runs $dir/
cp -r para_results $dir/
