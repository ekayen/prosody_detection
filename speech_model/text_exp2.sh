python3 train.py -c conf/cnn_lstm_cluster.yaml -l 2 -dr 0.2 -b 1500-e 300 &
python3 train.py -c conf/cnn_lstm_cluster.yaml -l 3 -dr 0.2 -b 1000-e 300 &
python3 train.py -c conf/cnn_lstm_cluster.yaml -l 3 -dr 0.2 -b 700-e 300 &
python3 train.py -c conf/cnn_lstm_cluster.yaml -l 2 -dr 0 -b 100-e 100 &
python3 train.py -c conf/cnn_lstm_cluster.yaml -l 2 -dr 0 -b 2000-e 100 &
python3 train.py -c conf/cnn_lstm_cluster.yaml -l 3 -dr 0.5 -b 1500-e 100 &
python3 train.py -c conf/cnn_lstm_cluster.yaml -l 2 -dr 0.2 -b 1000-e 100
