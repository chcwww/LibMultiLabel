LOAD="${1:-0}"
HOST=$2

if [ $LOAD -ne 0 ]; then
    # ssh-keygen -t rsa
    # ssh-copy-id USER@HOST
    scp -r ${HOST}:~/libmultilabel/data ./
else
    mkdir -p data/rcv1
    cd data/rcv1
    wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.txt.bz2
    wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_test.txt.bz2

    bzip2 -d *.bz2

    cd ../..

    mkdir -p data/EUR-Lex
    cd data/EUR-Lex
    wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex_raw_texts_train.txt.bz2
    wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex_raw_texts_test.txt.bz2

    bzip2 -d *.bz2

    cd ../..

    mkdir -p data/AmazonCat-13K
    cd data/AmazonCat-13K
    wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/AmazonCat-13K_raw_texts_train.txt.bz2 
    wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/AmazonCat-13K_raw_texts_test.txt.bz2

    bzip2 -d *.bz2

    cd ../..

    mkdir -p data/Wiki10-31K
    cd data/Wiki10-31K
    wget -O train.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/wiki10_31k_raw_texts_train.txt.bz2
    wget -O test.txt.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/wiki10_31k_raw_texts_test.txt.bz2

    bzip2 -d *.bz2
fi
