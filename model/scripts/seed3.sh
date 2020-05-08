python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold6.yaml -v 3000 -s $1 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold7.yaml -v 3000 -s $1 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold8.yaml -v 3000 -s $1 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold9.yaml -v 3000 -s $1