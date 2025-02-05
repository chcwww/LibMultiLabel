LOAD="${1:-0}"

# bash get_data.sh 0(pickle files from svm.csie.ntu.edu.tw) 1(libsvm format) other(libmultilabel format)

if [ $LOAD -eq 0 ]; then
    # ssh-keygen -t rsa
    # ssh-copy-id USER@HOST
    scp -r ${2}:~/libmultilabel/data ./
elif [ $LOAD -eq 1 ]; then
    mkdir -p data/rcv1
    cd data/rcv1
    wget -O train.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_train.svm.bz2
    wget -O test.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/rcv1_topics_combined_test.svm.bz2

    bzip2 -d *.bz2

    cd ../..

    mkdir -p data/EUR-Lex
    cd data/EUR-Lex
    wget -O train.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex_tfidf_train.svm.bz2
    wget -O test.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/eurlex_tfidf_test.svm.bz2

    bzip2 -d *.bz2

    cd ../..

    mkdir -p data/AmazonCat-13K
    cd data/AmazonCat-13K
    wget -O train.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/AmazonCat-13K_tfidf_train_ver1.svm.bz2 
    wget -O test.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/AmazonCat-13K_tfidf_test_ver1.svm.bz2

    bzip2 -d *.bz2

    cd ../..

    mkdir -p data/Wiki10-31K
    cd data/Wiki10-31K
    wget -O train.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/wiki10_31k_tfidf_train.svm.bz2
    wget -O test.svm.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/wiki10_31k_tfidf_test.svm.bz2

    bzip2 -d *.bz2
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
