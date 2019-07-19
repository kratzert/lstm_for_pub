#!/bin/bash

nseeds=12
firstseed=200

gpucount=-1
for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do

  gpucount=$(($gpucount + 1))
  gpu=$(($gpucount % 4))
  echo $seed $gpucount $gpu
  
  if [ "$1" = "lstm" ] 
  then

    outfile="reports/global_lstm_$2.$seed.out"

    if [ "$2" = "static" ] 
    then
      python3 main.py --gpu=$gpu --no_static=False --concat_static=True train > $outfile &
    elif [ "$2" = 'no_static' ]
    then
      python3 main.py --gpu=$gpu --no_static=True train > $outfile &
    else
      echo bad model choice
      exit
    fi

  else
    echo bad model choice
    exit
  fi

  if [ $gpu -eq 3 ]
  then
    wait
  fi

done
