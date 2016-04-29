train_data=train_files
al=0.001
time ./mf_train -f $train_data -m model_name --thread 6 --alpha $al --l2 0.005 --epoch 1s
