#python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold8.yaml -v 3000 &
#python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold9.yaml -v 3000 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/f2b_only0.yaml -v 1700 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/f2b_only1.yaml -v 1700 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/f2b_only2.yaml -v 1700 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/f2b_only3.yaml -v 1700 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/f2b_only4.yaml -v 1700 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/f2b_only5.yaml -v 1700
