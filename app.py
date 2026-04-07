import time
import sys
import psutil

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql import functions as F

from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

OPTIMIZED = sys.argv[1] == "True"

conf = SparkConf()
conf.set("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
conf.set("spark.executor.memory", "1500m")
conf.set("spark.driver.memory", "2g")
conf.set("spark.sql.shuffle.partitions", "8")
conf.set("spark.default.parallelism", "8")

spark = SparkSession.builder \
    .appName("telco_churn") \
    .master("spark://spark-master:7077") \
    .config(conf=conf) \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")

start_time = time.time()
process = psutil.Process()

print("START: read data")

df = spark.read.csv("hdfs:///data/train.csv", header=True, inferSchema=True)

print("DONE: read data")

print("START: preprocessing")

df = df.drop("CustomerID")

df = df.withColumn(
    "Churn",
    F.when(F.col("Churn") == "Yes", 1).otherwise(0)
)

categorical_cols = ["Gender", "Contract", "PaymentMethod"]
numeric_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges"]

df = df.fillna("Unknown")

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=numeric_cols
)

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]

feature_cols = numeric_cols + [f"{c}_idx" for c in categorical_cols]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

train_df, test_df = df.randomSplit([0.75, 0.25], seed=42)

pipeline = Pipeline(stages=[imputer] + indexers + [assembler])

pipeline_model = pipeline.fit(train_df)

train_df = pipeline_model.transform(train_df)
test_df = pipeline_model.transform(test_df)

print("DONE: preprocessing")

if OPTIMIZED:
    print("START: optimization")
    train_df = train_df.repartition(8).cache()
    test_df = test_df.repartition(8).cache()
    train_df.count()
    test_df.count()
    print("DONE: optimization")

print("START: training")

lr = LogisticRegression(featuresCol="features", labelCol="Churn")

param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol="Churn")

cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2
)

model = cv.fit(train_df)

print("DONE: training")

print("START: evaluation")

pred = model.transform(test_df)
auc = evaluator.evaluate(pred)

print("DONE: evaluation")

total_time = time.time() - start_time
ram_usage = process.memory_info().rss / (1024 * 1024)

print(f"{total_time:.2f},{ram_usage:.2f},{auc:.4f},{OPTIMIZED}")

spark.stop()