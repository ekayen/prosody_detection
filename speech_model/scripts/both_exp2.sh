python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.0001 -wd 0.0001 -l 3 -dr 0.5 -hid 128 -b 2000  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.0001 -wd 1e-05 -l 2 -dr 0 -hid 512 -b 10  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.01 -wd 1e-05 -l 2 -dr 0 -hid 512 -b 3000  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.0001 -wd 0 -l 2 -dr 0 -hid 128 -b 2500  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.001 -wd 1e-06 -l 2 -dr 0 -hid 512 -b 700  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.001 -wd 0.0001 -l 2 -dr 0 -hid 128 -b 70 