# NeuralModels
Storing work based on neural models


## Training models on Google Cloud ML.
The following steps will allow the training of TensorFlow models on Cloud ML. 
1. Write your TensorFlow code. 
2. Correctly package your code to a python package. 
3. Run package locally to test code.
4. Push code to Cloud ML to train model at scale.

### 1. Write your TensorFlow code. 
If you are new to TensorFlow, some preliminary tutorials can be found [here.](https://www.tensorflow.org/get_started/get_started) 

An example of a simple one-layer network for the XOR gate is provided. Feel free to alter/modify to understand the basics of TensorFlow. 

### 2. Correctly package your code to a python package.
To successfully train a model on Cloud ML, the code must be submitted as a python package. This is quite easy to achieve, we simply add a blank `__init__.py` file to the directory that our TensorFlow code is in. Secondly, we must structure our code from step 1 to run as a `tf.app`. To do this, append the following to your code, where `training()` is a function that encompasses your TensorFlow model. 


```python 
def main(_):
    training()

if __name__ == "__main__": 
    tf.app.run() 
```

### 3. Run package locally to test code.
Before you submit your job to Cloud ML, it is a good idea to run it locally. First, you need to set up the [Google Cloud SDK](https://cloud.google.com/ml-engine/docs/quickstarts/command-line) in your console which can be found [here.](https://cloud.google.com/ml-engine/docs/quickstarts/command-line) Once you have successfully set the Google Cloud SDK, you are ready to test your code locally. To test the code locally, `cd` into the root of the repository. If you `ls` the directory should contain the folder `train`. Run the following to train the model:

``` 
gcloud ml-engine local train  \
--package-path=train \
--module-name=train.XOR 
```

You should see logs of the model training in your console.

### 4. Push code to Cloud ML to train model at scale.
Now you have written your TensorFlow code, packaged it and run it locally it is time to run it at scale on Cloud ML. 
Again, if you have not done so already, you need to set up the [Google Cloud SDK](https://cloud.google.com/ml-engine/docs/quickstarts/command-line) in your console which can be found [here.](https://cloud.google.com/ml-engine/docs/quickstarts/command-line)

To run your code in Cloud ML, you will need to push it to a staging bucket. This will store your code, acting as a reference for the Cloud ML service. To do this, go to your Google Cloud Console, navigate to `Storage`. Inside `Storage` click the `CREATE BUCKET`. Name the bucket `staging-bucket-xor` and accept the default settings. 

Set the following environment variables by running the following:

``` 
export STAGING_BUCKET=gs://staging-bucket-xor
export JOB_NAME=XORgate
```

Once you have created these environment variables, you are now ready to push your python package to Cloud ML. From the root directory, run the following command:

``` 
gcloud ml-engine jobs submit training ${JOB_NAME} \
--package-path=train \
--staging-bucket="${STAGING_BUCKET}" \
--region us-central1 \
--module-name=train.XOR 
```

Running this command should submit your code to Cloud ML. If the job was successfully submitted, you should get the following response:

`Job [XORgate] submitted successfully.`


Once you have successfully submitted your job, it will become visible in the [Google Cloud ML console.](https://console.cloud.google.com/mlengine) It should be called `XORgate`. NOTE: Cloud ML can take a while to get provision resources and complete the training. 
