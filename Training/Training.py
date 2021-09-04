#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer


# In[2]:


# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
my_region = boto3.session.Session().region_name # set the region of the instance
print(my_region)


# In[3]:


# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")


# In[4]:


print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")


# In[5]:


# Create the S3 bucket
bucket_name = 'trialbucket001' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)


# In[6]:


# set an output path where the trained model will be saved
output_path = 's3://{}/{}/output'.format(bucket_name,prefix)
print(output_path)


# ### Downloading the Dataset and storing in s3

# In[7]:


try:
  urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
  print('Success: downloaded bank_clean.csv.')
except Exception as e:
  print('Data load error: ',e)

try:
  model_data = pd.read_csv('./bank_clean.csv',index_col=0)
  print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)


# In[8]:


model_data.head()


# In[9]:


# split into train and test data
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)


# ### Saving Train data and Test data in bucket

# ### Train Data

# In[10]:


pd.concat([train_data['y_yes'], 
           train_data.drop(['y_no', 'y_yes'], 
                           axis=1)], axis=1).to_csv('train.csv', 
                                                    index=False, 
                                                    header=False)

# saving train data into s3 bucket.
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')

# save it in same instance
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# ### Test Data

# In[11]:


pd.concat([test_data['y_yes'], 
           test_data.drop(['y_no', 'y_yes'], 
                           axis=1)], axis=1).to_csv('test.csv', 
                                                    index=False, 
                                                    header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')


# ## create an instance of the XGBoost model (an estimator), and define the model’s hyperparameters.
# 
# # Building and training XGboost algo

# In[12]:


sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(xgboost_container,
                                    role, 
                                    instance_count=1, 
                                    instance_type='ml.m4.xlarge',
                                    volume_size = 5,
                                    output_path='s3://{}/{}/output'.format(bucket_name, prefix),
                                    sagemaker_session=sess,
                                    use_spot_instances = True,
                                    max_run=300,
                                    max_wait = 600
                                   )


# In[13]:


xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='binary:logistic',
                        num_round=100)


# In[14]:


xgb.fit({'train': s3_input_train, 'validation': s3_input_test})


# # Step 4. Deploy the model

# In[15]:


xgb_predictor = xgb.deploy(initial_instance_count=1,
                           instance_type='ml.m4.xlarge')


# # Step 5. Prediction on Test Data.

# In[16]:


from sagemaker.serializers import CSVSerializer

test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.serializer = CSVSerializer() # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)


# In[17]:


predictions_array


# # Step 6: Cross Validation
# 
# https://www.healthcare.uiowa.edu/path_handbook/appendix/chem/pred_value_theory.html
# 
# PREDICTIVE VALUE:
# 
# The predictive value of a test is a measure (%) of the times that the value (positive or negative) is the true value, i.e. the percent of all positive tests that are true positives is the Positive Predictive Value.
# 
# __TP___ X 100 = Predictive Value of a Positive Result (%)
# TP + FP
# 
# __TN___ X 100 = Predictive Value Negative Result (%)
# FN + TN
# 
# 
# http://www.academicos.ccadet.unam.mx/jorge.marquez/cursos/Instrumentacion/FalsePositive_TrueNegative_etc.pdf
# 
# 
# The Positive Predictive Value, also known as the precision rate, or the post-test
# probability of a disease. It is the proportion of patients with positive test results
# who are correctly diagnosed. It is the most important measure of a diagnostic
# method as it reflects the probability that a positive test reflects the underlying
# condition being tested for. Its value does however depend on the prevalence of
# the disease, which may vary. In terms of the latter definitions, we have:
# ( )( ) TP PPV ( )( ) (1 )(1 ) TP+FP
# sensitivity prevalence
# sensitivity prevalence specificity prevalence
#  
#   
# and, similarly, we have the Negative Predictive Value:
# TN NPV TN+FN

# In[19]:


cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))


# # Step 6. Clean up

# In[20]:


xgb_predictor.delete_endpoint(delete_endpoint_config=True)


# In[21]:


bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()


# In[ ]:




