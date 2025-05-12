dir=$1
mkdir $dir
mv *.dat $dir/
mv memory_profile $dir/
mv para_log $dir/
mv runs $dir/
cp -r para_results $dir/
