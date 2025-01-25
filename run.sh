datasets=(rcv1 EUR-Lex Wiki10-31K AmazonCat-13K)
linear_techs=(tree 1vsrest)
branches=(master no_parallel ovr_thread sep_ovr_thread)

LOG=$1

for linear_tech in "${linear_techs[@]}"; do 
    mkdir -p para_log/${linear_tech}  memory_profile/${linear_tech}
done

for linear_tech in "${linear_techs[@]}"; do 
    for data in "${datasets[@]}"; do
        for branch in "${branches[@]}"; do
            exp_id=${linear_tech}/${data}--${linear_tech}--${branch}
            git checkout ${branch}

            mprof run -M -C python -X faulthandler main.py \
            --linear --liblinear_options "-s 1 -B 1 -e 0.0001 -q" \
            --data_format txt --training_file data/${data}/train.txt \
            --linear_technique ${linear_tech} --seed 1337\
            --result_dir runs/${exp_id} \
            > para_log/${exp_id}.log 2>&1

            mprof plot -o memory_profile/${exp_id}.png
            ps -u chcwww | grep python | grep -oE "[0-9]+ pts" | grep -oE "[0-9]+" | xargs kill -9
        done
    done
done

bash move_log.sh ${LOG}
git checkout master