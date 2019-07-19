#!/bin/bash

nsplits=12
nseeds=10
firstseed=300

gpucount=-1
for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do

  python3 main.py --n_splits=$nsplits --seed=$seed create_splits 
  wait

  for ((split = 0 ; split < $nsplits ; split++ )); do  

    gpucount=$(($gpucount + 1))
    gpu=$(($gpucount % 3))
    echo $seed $gpucount $gpu

    if [ "$1" = "lstm" ]
    then

      outfile="reports/pub_lstm_extended_nldas.$seed.$split.out"
      python3 main.py --gpu=$gpu --no_static=False --concat_static=True --split=$split --split_file="data/kfold_splits_seed$seed.p" train > $outfile &

    elif [ "$1" = 'ealstm' ]
    then

      outfile="reports/pub_ealstm_extended_nldas.$seed.$split.out"
      python3 main.py --gpu=$gpu --split=$split --split_file="data/kfold_splits_seed$seed.p" train > $outfile &

    else
      echo bad model choice
      exit
    fi

    if [ $gpu -eq 2 ]
    then
      wait
    fi

  done
done

