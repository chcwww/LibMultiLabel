if [ $# -eq 0 ]; then
    echo "Need an argument for log directory name."
    exit
fi

datasets=(rcv1 EUR-Lex Wiki10-31K AmazonCat-13K)
datasets=(AmazonCat-13K)

for data in "${datasets[@]}"; do
    python main_inference.py --data_format svm \
    --training_file data/${data}/train.svm --test_file data/${data}/test.svm \
    --data_name ${data} --result_dir $1 > $1/para_log/${data}_eval.log 2>&1
    
    ps -u chcwww | grep python | grep -oE "[0-9]+ pts" | grep -oE "[0-9]+" | xargs kill -9
done

# bash inference.sh log_dir
            
