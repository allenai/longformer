#!/bin/bash

export SCRIPTS=$(beaker dataset create -q .)
export INPUT_DATASET_ID="ds_6r0phxc5fiap"
export RESULT_SAVE_DIR="/runs"
export RESULT_SAVE_PREFIX="test"
export ARGS=""
export GPU_COUNT=1
export CPU_COUNT=6
copy=("$@")
for i in "${!copy[@]}"
do
  if [[ "${copy[$i]}" = "--save_dir" ]]
  then
    export RESULT_SAVE_DIR="${copy[$i+1]}"
  fi

  if [[ "${copy[$i]}" = "--input_dir" ]]
  then
    export INPUT_DATASET_ID=$(beaker dataset create -q ${copy[$i+1]})
    copy[$i+1]="/data"
  fi

  if [[ "${copy[$i]}" = "--save_prefix" ]]
  then
    export RESULT_SAVE_PREFIX="${copy[$i+1]}"
  fi

  if [[ "${copy[$i]}" = "--num_workers" ]]
  then
    export CPU_COUNT="${copy[$i+1]}"
  fi

  if [[ "${copy[$i]}" = "--gpu_count" ]]
  then
    export GPU_COUNT="${copy[$i+1]}"
  fi
  ARGS="$ARGS ${copy[$i]}"
done

# If an input dataset was not specified, use the default
if [[ "ds_6r0phxc5fiap" = $INPUT_DATASET_ID ]]
then
  ARGS="$ARGS --input_dir /data"
fi

echo $ARGS

export RESULT_PATH=$RESULT_SAVE_DIR/$RESULT_SAVE_PREFIX

beaker experiment create -f experiment.yml
