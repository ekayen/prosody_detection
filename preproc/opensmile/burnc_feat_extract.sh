#!/bin/bash

burnc_dir='burnc'

for f in $burnc_dir/*.wav; do
    if [[ ! -f "${f%.*}.csv" ]]; then
	./SMILExtract -C config/IS13_ComParE.edit.conf -I $f -csvoutput "${f%.*}.is13.csv"
    fi
done


