if [ $# -eq 0 ]; then
    echo "Need an argument for log directory name."
    exit
fi

branch=timing_exp
datasets=(rcv1 AmazonCat-13K EUR-Lex Wiki10-31K)
linear_techs=(1vsrest tree)
num_threads=(1 2 4 8)

datasets=(rcv1)

test_data=""
if [ $# -eq 3 ]; then
    # bash run.sh log_dir linear_technique dataset_name
    linear_techs=($2)
    datasets=($3)
    # test_data="--test_file data/${data}/test.svm"
fi

mkdir -p para_log  memory_profile para_results

for linear_tech in "${linear_techs[@]}"; do 
    for data in "${datasets[@]}"; do
        for num_thread in "${num_threads[@]}"; do
            exp_id=${linear_tech}--${data}--${linear_tech}--${num_thread}

            NO_PARA=0 NO_COPY=0 mprof run -M -C python -X faulthandler main.py \
            --num_threads ${num_thread} --training_only --dict_output_path para_results/${exp_id}.json \
            --linear --liblinear_options "-m ${num_thread}" \
            --data_format svm --training_file data/${data}/train.svm ${test_data}\
            --linear_technique ${linear_tech} --seed 1337 \
            --result_dir runs/${exp_id} \
            > para_log/${exp_id}.log 2>&1

            mprof plot -o memory_profile/${exp_id}.png
            ps -u chcwww | grep python | grep -oE "[0-9]+ pts" | grep -oE "[0-9]+" | xargs kill -9
        done
    done
done

bash move_log.sh $1

# python -X faulthandler main.py --linear --liblinear_options "-s 1 -B 1 -e 0.0001 -q" --data_format txt --training_file data/rcv1/train.txt --linear_technique 1vsrest --seed 1337 --result_dir runs/1vsrest
# | tee -a para_log/${exp_id}.log
