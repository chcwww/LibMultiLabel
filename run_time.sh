if [ $# -eq 0 ]; then
    echo "Need an argument for log directory name."
    exit
fi

branch=timing_exp
datasets=(rcv1 AmazonCat-13K EUR-Lex Wiki10-31K)
linear_techs=(1vsrest tree)
num_threads=(1 2 4 8 0)

datasets=(rcv1 EUR-Lex)

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
            no_para=0
            no_copy=0
            stech="linear_tech=${linear_tech}"
            sdata="data=${data}"
            snum_thread="num_thread=${num_thread}"
            sno_para="no_para=${no_para}"
            sno_copy="no_copy=${no_copy}"
            sname="name=${1}"
            exp_id=$stech--$sdata--$snum_thread--$sno_para--$sno_copy--$sname

            liblinear_options="-m ${num_thread}"
            if [ $num_thread -eq 0 ]; then
                no_para=1
                liblinear_options="-s 1"
            fi

            NO_PARA=${no_para} NO_COPY=${no_copy} python main.py \
            --num_threads ${num_thread} --training_only --dict_output_path para_results/${exp_id}.json \
            --linear --liblinear_options ${liblinear_options} \
            --data_format svm --training_file data/${data}/train.svm ${test_data}\
            --linear_technique ${linear_tech} --seed 1337 \
            --result_dir runs/${exp_id} \
            > para_log/${exp_id}.log 2>&1

            # mprof plot -o memory_profile/${exp_id}.png
            # pkill -9 python
        done
    done
done

bash move_log.sh $1

# mprof run -M -C python -X faulthandler main.py --linear --liblinear_options "-s 1 -B 1 -e 0.0001 -q" --data_format txt --training_file data/rcv1/train.txt --linear_technique 1vsrest --seed 1337 --result_dir runs/1vsrest
# | tee -a para_log/${exp_id}.log
