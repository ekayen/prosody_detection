echo "preprocessing"
python3 swbd_preproc.py
echo "splitting"
python3 swbd_gen_splits.py
echo "stopwords"
./run_stopwords.sh ../data/swbd/splits
echo "done!"
