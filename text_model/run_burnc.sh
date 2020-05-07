#!/bin/bash

for f in run_burnc/configs/*
do
  echo $f
  nice python model.py $f
  echo "done"
done