#!/bin/bash

DATA_PATH="data/train.csv"
FILENAME="train.csv"
SPARK_APP="app.py"
HDFS_PATH="/data"

NODES=1
OPTIMIZED=False

while [[ $# -gt 0 ]]
do
  case "$1" in
    --nodes)
      NODES="$2"
      shift 2
      ;;
    --optimized)
      OPTIMIZED="$2"
      shift 2
      ;;
  esac
done

if [[ $NODES -eq 1 ]]
then
  COMPOSE_FILE="docker-compose-1.yml"
else
  COMPOSE_FILE="docker-compose-3.yml"
fi

docker compose -f $COMPOSE_FILE down --remove-orphans
docker compose -f $COMPOSE_FILE up -d

sleep 20

docker cp "$DATA_PATH" namenode:/

docker exec namenode hdfs dfs -mkdir -p $HDFS_PATH
docker exec namenode hdfs dfs -put -f "/$FILENAME" "$HDFS_PATH/$FILENAME"

docker cp "$SPARK_APP" spark-master:/tmp/

docker exec spark-master apk add --no-cache python3 py3-pip py3-numpy py3-psutil
docker exec spark-worker-1 apk add --no-cache python3 py3-pip py3-numpy py3-psutil

mkdir -p logs

if [ ! -f logs/all_results.csv ]; then
  echo "nodes,optimized,time,ram,auc" > logs/all_results.csv
fi

OUTPUT_FILE=logs/tmp_output.txt

docker exec spark-master /spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /tmp/$SPARK_APP $OPTIMIZED > $OUTPUT_FILE 2>&1

RESULT=$(grep -E '^[0-9]+\.[0-9]+,[0-9]+\.[0-9]+,[0-9]+\.[0-9]+,(True|False)$' $OUTPUT_FILE | tail -n 1)

TIME=$(echo $RESULT | cut -d',' -f1)
RAM=$(echo $RESULT | cut -d',' -f2)
AUC=$(echo $RESULT | cut -d',' -f3)
OPT=$(echo $RESULT | cut -d',' -f4)

echo "$NODES,$OPT,$TIME,$RAM,$AUC" >> logs/all_results.csv

mv $OUTPUT_FILE logs/log_${NODES}_${OPTIMIZED}.log

docker compose -f $COMPOSE_FILE down