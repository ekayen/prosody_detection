python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.0001 -wd 0.01 -l 2 -dr 0.2 -hid 128 -b 10  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.001 -wd 0.01 -l 3 -dr 0.2 -hid 512 -b 1000  
