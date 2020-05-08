python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.001 -wd 0.001 -l 2 -dr 0 -hid 512 -b 700  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.0001 -wd 1e-06 -l 2 -dr 0.2 -hid 512 -b 1000  
