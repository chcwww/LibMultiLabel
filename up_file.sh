for folder in $1/runs/*; do
    cd $folder
    for subfolder in ./*; do
        cd $subfolder
        mv ./*/linear_pipeline.pickle ./
        cd ..
    done
    cd ../../..
done
# bash up_file.sh log_dir