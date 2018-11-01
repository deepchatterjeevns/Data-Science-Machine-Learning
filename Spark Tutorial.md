<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/>
</a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>
# CIS5560 Term Project Tutorial

#### Authors: Hai Anh Le, Neha Gupta, Maria Boldina
#### Instructor:  [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)
#### Date: 05/18/2017
#
### Import Spark SQL and Spark ML Libraries
```python
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import date_format
import pyspark.sql.functions as func
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
```

### Load Source Data

The data for this exercise is provided as a CSV file containing details of users click. The data includes specific characteristics for each user, as well as a column indicating how many user download the app or not.

```python
train_sampleSchema = StructType([
  StructField("ip", IntegerType(), False),
  StructField("app", IntegerType(), False),
  StructField("device", IntegerType(), False),
  StructField("os", IntegerType(), False),
  StructField("channel", IntegerType(), False),
  StructField("clicktime", TimestampType (), False),
  StructField("attributed", TimestampType(), False),
  StructField("is_attributed", IntegerType(), False),
])
```
### Read csv file from DBFS (Databricks File Systems)

1.  After train_sample_1G.csv file is added to the data of the left frame, create a table using the UI, especially, "Upload File"
2.  Click "Preview Table to view the table" and Select the option as train_sample_1G.csv has a header as the first row: "First line is header"
3.  Change the data type of the table columns as shown in train_sampleSchema of the above cell
4.  When you click on create table button, remember the table name, for example,  _train_sample_1G_

```python
%fs ls /FileStore/tables/train_sample_1G.csv
```

### Create a dataframe from the table, using Spark SQL
```python
df = spark.sql("SELECT * FROM train_sample_1G_csv")
```
### Counting the counts of 0's and 1's from is_attributed column to check how many users download the app
```python
count = df.groupBy('is_attributed').count()
```

### Prepare the Data
Most modeling begins with exhaustive exploration and preparation of the data. In this example, the data has been cleaned for you. You will simply select a subset of columns to use as  _features_  and create a Boolean  _label_  field named  **label**  with the value  **1**  for users who downloaded the app, or  **0**  for the users who did not download the app.
```python
train = df
```

### Building the features

### Feature -1: Prepare time based feature by extracting day of the week and hour of the day from the click time
```python
train_with_day_of_week = train.withColumn('day_of_week_number',date_format('click_time', 'u').cast('integer')).withColumn('hour_of_day', date_format('click_time', 'H').cast('integer'))
```

### Feature -2: Prepare feature by grouping clicks by combination of (Ip, Day_of_week_number and Hour)
```python
grpd_by_ip_day_hr = train_with_day_of_week.groupBy('ip', 'day_of_week_number', 'hour_of_day').agg(func.count(func.lit(1)).alias("count_by_ip_day_hour"))
```

### Adding Features back to the original dataset
```python
joined_1 = train_with_day_of_week.join(grpd_by_ip_day_hr, ['ip','day_of_week_number','hour_of_day'], "leftouter")
train_with_day_of_week.unpersist()
grpd_by_ip_day_hr.unpersist()
```

### Feature -3: Prepare feature by grouping clicks by combination of (Ip, App, Operating System, Day_of_week_number and Hour)
```python
grpd_by_ip_app_os_day_hr = train_with_day_of_week.groupBy('ip', 'app','os','day_of_week_number', 'hour_of_day').agg(func.count(func.lit(1)).alias("count_by_ip_app_os_day_hour"))
```

### Adding Features back to the original dataset
```python
joined_2 = joined_1.join(grpd_by_ip_app_os_day_hr, ['ip','app','os','day_of_week_number','hour_of_day'], "leftouter")
joined_1.unpersist()
grpd_by_ip_app_os_day_hr.unpersist()
```

### Feature -4 : Prepare feature by grouping clicks by combination of (App, Day_of_week_number and Hour)
```python
grpd_by_app_day_hr = train_with_day_of_week.groupBy('app','day_of_week_number', 'hour_of_day').agg(func.count(func.lit(1)).alias("grpd_by_app_day_hr"))
```

### Adding Features back to the original dataset
```python
joined_3 = joined_2.join(grpd_by_app_day_hr, ['app','day_of_week_number','hour_of_day'], "leftouter")
joined_2.unpersist()
grpd_by_app_day_hr.unpersist()
```

### Feature -5 : Prepare feature by grouping clicks by combination of (Ip, App, Device and Operating System)
```python
grpd_by_ip_app_dev_os = train_with_day_of_week.groupBy('ip','app','device', 'os').agg(func.count(func.lit(1)).alias("grpd_by_ip_app_dev_os"))
```

### Adding Features back to the original dataset
```python
joined_4 = joined_3.join(grpd_by_ip_app_dev_os, ['ip','app','device','os'], "leftouter")
joined_3.unpersist()
grpd_by_ip_app_dev_os.unpersist()
```
### Feature -6 : Prepare feature by grouping clicks by combination of (Ip, Device and Operating System)
```python
grpd_by_ip_dev_os = train_with_day_of_week.groupBy('ip','device', 'os').agg(func.count(func.lit(1)).alias("grpd_by_ip_dev_os"))
```
### Adding Features back to the original dataset
```python
joined_5 = joined_4.join(grpd_by_ip_dev_os, ['ip','device','os'], "leftouter")
joined_4.unpersist()
grpd_by_ip_dev_os.unpersist()
```
### Consolidating the data and renaming the target column name (is_attributed) to label
```python
data = joined_5.select('ip', 'device', 'os', 'app', 'day_of_week_number', 'hour_of_day','channel','count_by_ip_day_hour','count_by_ip_app_os_day_hour','grpd_by_app_day_hr','grpd_by_ip_app_dev_os','grpd_by_ip_dev_os', (col("is_attributed").cast("Int").alias("label")))
```
### Split the Data

It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing.
```python
splits = data.randomSplit([0.7, 0.3],4272)
trainingData = splits[0]
trainingData.cache()
print trainingData.count() # explicitly calling count to cache the training data in memory
testingData = splits[1].withColumnRenamed("label", "trueLabel")
print testingData.count() # explicitly calling count to cache the testing data in memory
testingData.cache()
```
### Define the Pipeline

A predictive model often requires multiple stages of feature preparation. For example, it is common when using some algorithms to distingish between continuous features (which have a calculable numeric value) and categorical features (which are numeric representations of discrete categories). It is also common to  _normalize_continuous numeric features to use a common scale (for example, by scaling all numbers to a proportinal decimal value between 0 and 1).

A pipeline consists of a a series of  _transformer_  and  _estimator_  stages that typically prepare a DataFrame for modeling and then train a predictive model. In this case, you will create a pipeline with two stages:

-   A  **StringIndexer**  estimator that converts string values to indexes for categorical features
-   A  **VectorAssembler**  that combines categorical features into a single vector
-   A  **VectorIndexer**  that creates indexes for a vector of categorical features
-   A  **VectorAssembler**  that creates a vector of continuous numeric features
-   A  **MinMaxScaler**  that normalizes continuous numeric features
-   A  **VectorAssembler**  that creates a vector of categorical and continuous features
-   A  **DecisionTreeClassifier**  that trains a classification model.
```python
va = VectorAssembler(inputCols = ['ip', 'device', 'os', 'app', 'day_of_week_number', 'hour_of_day','channel','count_by_ip_day_hour','count_by_ip_app_os_day_hour','grpd_by_app_day_hr','grpd_by_ip_app_dev_os','grpd_by_ip_dev_os'], outputCol="features")
vi = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4)
```

### Assigning pipeline variables for Decision Tree Classifier Model

The Decision Trees algorithm is popular because it handles categorical data and works out of the box with multiclass classification tasks
```python
dt = DecisionTreeClassifier(labelCol="label", featuresCol="indexedFeatures", maxDepth=3)
```

### Assigning pipeline variables for Random Forest Classifier Model

Random Forests uses an ensemble of trees to improve model accuracy. You can read more about Random Forest from the classification and regression section of MLlib Programming Guide.
```python
rf = RandomForestClassifier(labelCol="label", featuresCol="indexedFeatures")
```

### Assigning Pipeline
```python
dtp = Pipeline(stages=[va, vi, dt])
rfp = Pipeline(stages=[va, vi, rf])
model = []
```

### Tune Parameters

You can tune parameters to find the best model for your data. A simple way to do this is to use  **TrainValidationSplit**  to evaluate each combination of parameters defined in a  **ParameterGrid**  against a subset of the training data in order to find the best performing parameters.
#### Regularization

is a way of avoiding Imbalances in the way that the data is trained against the training data so that the model ends up being over fit to the training data. In other words It works really well with the training data but it doesn't generalize well with other data. That we can use a  **regularization parameter**  to vary the way that the model balances that way.

#### Training ratio of 0.8

it's going to use 80% of the the data that it's got in its training set to train the model and then the remaining 20% is going to use to validate the trained model.

In  **ParamGridBuilder**, all possible combinations are generated from regParam, maxIter, threshold. So it is going to try each combination of the parameters with 80% of the the data to train the model and 20% to to validate it.

### ParamGridBuilder for Decision Tree Classifier Model
```python
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [1, 2, 6, 10])
             .addGrid(dt.maxBins, [20, 40, 80])
             .build())
```

### Building and Training Decision Tree Classifier Model using Train Validation Split
```python
dt_tvs = TrainValidationSplit(estimator=dtp, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
model.insert(0, dt_tvs.fit(trainingData))
```

### ParamGridBuilder for Random Forest Classifier Model
```python
paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())
```

### Building and Training Random Forest Classifier Model using Train Validation Split
```python
rf_tvs = TrainValidationSplit(estimator=rfp, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
model.insert(1, rf_tvs.fit(trainingData))
```

### Test the Model

Now you're ready to use the  **transform**  method of the model to generate some predictions. You can use this approach to predict if the app will be downloaded where the label is is-attributed. Also in this case you are using the test data which includes a known true label value, so you can compare the predicted number of clicks which actually led to app being downloaded.

%md ### Test the Pipeline Model The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the  **test**  DataFrame using the pipeline to generate label predictions.
```python
prediction = [] 
predicted = []
for i in range(2):
  prediction.insert(i, model[i].transform(testingData))
  predicted.insert(i, prediction[i].select("features", "prediction", "probability", "trueLabel"))
  predicted[i].show(15)
```

### Visualization of Truelable and Prediction for Decision Tree Classifier Model
```python
display(predicted[0])
```


### Visualization of Truelable and Prediction for Random Forest Classifier Model
```python
display(predicted[1])
```

### Compute Confusion Matrix Metrics: For Decision Tree Classifier

Classifiers are typically evaluated by creating a  _confusion matrix_, which indicates the number of:

-   True Positives
-   True Negatives
-   False Positives
-   False Negatives

From these core measures, other evaluation metrics such as  _precision_  and  _recall_  can be calculated.
```python
tp = float(predicted[0].filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted[0].filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted[0].filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted[0].filter("prediction == 0.0 AND truelabel == 1").count())
dt_metrics = spark.createDataFrame([
 ("TP", tp),
("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
dt_metrics.show()
```

### Visualization of Compute Confusion Matrix Metrics of Decision Tree Classifier
```python
display(dt_metrics)
```


### Compute Confusion Matrix Metrics: For Random Forest Classifier

Classifiers are typically evaluated by creating a  _confusion matrix_, which indicates the number of:

-   True Positives
-   True Negatives
-   False Positives
-   False Negatives

From these core measures, other evaluation metrics such as  _precision_  and  _recall_  can be calculated.
```python
tp = float(predicted[1].filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted[1].filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted[1].filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted[1].filter("prediction == 0.0 AND truelabel == 1").count())
rf_metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
rf_metrics.show()
```


### Visualization of Compute Confusion Matrix Metrics of Random Forest Classifier

```python
display(rf_metrics)
```


### Calculating Area Under Curve For Decision Tree Classifier
```python
dtree_evaluator =  BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
dt_auc = dtree_evaluator.evaluate(prediction[0])
print "AUC for Descision Tree Classifier  "," = ", dt_auc
```

### Calculating Area Under Curve For Random Forest Classifier
```python
rf_evaluator =  BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
rf_auc = rf_evaluator.evaluate(prediction[1])
print "AUC for Random Forest Classifier  "," = ", rf_auc
```


### Table of Area Under Curve For Decision Tree Classifier and Random Forest Classifier
```python
display(spark.createDataFrame([ 

("Decision Tree Classifier", dt_auc), 

("Random Forest Classifier", rf_auc)], ["Algorithm", "Area Under Curve"]))
```

### Creating Evaluator for Calculating Root Mean Square Error
```python
evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
```

### Calculating Root Mean Square Error for Decision Tree Classifier
```python
dtree_rmse = evaluator.evaluate(prediction[0])

print "Root Mean Square Error (RMSE) for Decision Tree Classifier:", dtree_rmse
```


### Calculating Root Mean Square Error for Random Forest Classifier
```python
rf_rmse = evaluator.evaluate(prediction[1])
print "Root Mean Square Error (RMSE) for Random Forest Classifier:", rf_rmse
```

### Table of Root Mean Square Error for Decision Tree Classifier and Random Forest Classifier
```python
display(spark.createDataFrame([ 
 ("Decision Tree Classifier", dtree_rmse), 
 ("Random Forest Classifier", rf_rmse)], ["Algorithm", "Root Mean Square Error"]))
```

### References:

1.  [DataBricks Guide For Spark ML](https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html)
2.  [DataSet Link for Spark ML tutorial](https://drive.google.com/file/d/1NiR9dYtEMZnWIMAw-FBEvP_MBcjrcRA4/)
