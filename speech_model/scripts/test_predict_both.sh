python predict_test.py -c conf/cnn_lstm_best.yaml -m results/refactor/crossval_seeds/both_crossval_seeds_tenfold0_s$1_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt -d ../data/burnc/splits/tenfold0.yaml -o both_test_$1 -rs $1 &
python predict_test.py -c conf/cnn_lstm_best.yaml -m results/refactor/crossval_seeds/both_crossval_seeds_tenfold1_s$1_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt -d ../data/burnc/splits/tenfold1.yaml -o both_test_$1 -rs $1 &
python predict_test.py -c conf/cnn_lstm_best.yaml -m results/refactor/crossval_seeds/both_crossval_seeds_tenfold2_s$1_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt -d ../data/burnc/splits/tenfold2.yaml -o both_test_$1 -rs $1 &
python predict_test.py -c conf/cnn_lstm_best.yaml -m results/refactor/crossval_seeds/both_crossval_seeds_tenfold3_s$1_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt -d ../data/burnc/splits/tenfold3.yaml -o both_test_$1 -rs $1 &
python predict_test.py -c conf/cnn_lstm_best.yaml -m results/refactor/crossval_seeds/both_crossval_seeds_tenfold4_s$1_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt -d ../data/burnc/splits/tenfold4.yaml -o both_test_$1 -rs $1 &
python predict_test.py -c conf/cnn_lstm_best.yaml -m results/refactor/crossval_seeds/both_crossval_seeds_tenfold5_s$1_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt -d ../data/burnc/splits/tenfold5.yaml -o both_test_$1 -rs $1 &
python predict_test.py -c conf/cnn_lstm_best.yaml -m results/refactor/crossval_seeds/both_crossval_seeds_tenfold6_s$1_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt -d ../data/burnc/splits/tenfold6.yaml -o both_test_$1 -rs $1 &
python predict_test.py -c conf/cnn_lstm_best.yaml -m results/refactor/crossval_seeds/both_crossval_seeds_tenfold7_s$1_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt -d ../data/burnc/splits/tenfold7.yaml -o both_test_$1 -rs $1 &
python predict_test.py -c conf/cnn_lstm_best.yaml -m results/refactor/crossval_seeds/both_crossval_seeds_tenfold8_s$1_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt -d ../data/burnc/splits/tenfold8.yaml -o both_test_$1 -rs $1 &
python predict_test.py -c conf/cnn_lstm_best.yaml -m results/refactor/crossval_seeds/both_crossval_seeds_tenfold9_s$1_cnn3_lstm2_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pt -d ../data/burnc/splits/tenfold9.yaml -o both_test_$1 -rs $1