python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.01 -wd 1e-06 -l 2 -dr 0.5 -hid 256 -b 700  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.0001 -wd 1e-06 -l 2 -dr 0.2 -hid 128 -b 10  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.01 -wd 0 -l 3 -dr 0.2 -hid 128 -b 2000  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.0001 -wd 1e-05 -l 3 -dr 0.2 -hid 512 -b 100  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.001 -wd 0.0001 -l 2 -dr 0.2 -hid 128 -b 2000  &
python3 train.py -c conf/cnn_lstm_pros.yaml -lr 0.001 -wd 0.0001 -l 2 -dr 0.5 -hid 256 -b 2000 