python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold0.yaml -v 3000 -s $1 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold1.yaml -v 3000 -s $1 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold2.yaml -v 3000 -s $1 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold3.yaml -v 3000 -s $1 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold4.yaml -v 3000 -s $1 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold5.yaml -v 3000 -s $1 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold6.yaml -v 3000 -s $1 &
python3 train.py -c conf/cnn_lstm_best.yaml  -d ../data/burnc/splits/tenfold7.yaml -v 3000 -s $1