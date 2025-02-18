if [ $# -eq 0 ]; then
    echo "Need an argument for log directory name."
    exit
fi

datasets=(rcv1 AmazonCat-13K EUR-Lex Wiki10-31K)
linear_techs=(1vsrest tree)
branches=(ovr_thread sep_ovr_thread master no_parallel)

test_data=""
if [ $# -eq 3 ]; then
    # bash run.sh log_dir linear_technique dataset_name
    linear_techs=($2)
    datasets=($3)
    # test_data="--test_file data/${data}/test.svm"
fi

for linear_tech in "${linear_techs[@]}"; do 
    mkdir -p para_log/${linear_tech}  memory_profile/${linear_tech}
done

for linear_tech in "${linear_techs[@]}"; do 
    for data in "${datasets[@]}"; do
        for branch in "${branches[@]}"; do
            exp_id=${linear_tech}/${data}--${linear_tech}--${branch}
            git checkout ${branch}

            mprof run -M -C python -X faulthandler main.py \
            --linear --liblinear_options "" \
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
git checkout master

bash up_file.sh $1
# bash inference.sh $1

# python -X faulthandler main.py --linear --liblinear_options "-s 1 -B 1 -e 0.0001 -q" --data_format txt --training_file data/rcv1/train.txt --linear_technique 1vsrest --seed 1337 --result_dir runs/1vsrest
# | tee -a para_log/${exp_id}.log
