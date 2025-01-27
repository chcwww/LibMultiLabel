if [ $# -eq 0 ]; then
    echo "Need an argument for log directory name."
    exit
fi

prefix="."
offset=2
if [ $# -eq 2 ]; then
    prefix=$2
    offset=3
fi

log_dir=$prefix/para_log/$1

output_file=$prefix/para_log/${1}_time.log

echo "data, tech, time" > "$output_file"

for log_file in ${log_dir}/*.log; do
    seconds=$(awk -F'in ' '/linear_train/ && /finished in/ {print $2+0}' "$log_file")
    log_name=(${log_file//// }) # //;/
    exp_name=(${log_name[3]//--/ })
    tech_name=(${exp_name[2]//./ })
    
    echo "${exp_name[0]}, ${tech_name[0]}, $seconds" >> "$output_file"
done