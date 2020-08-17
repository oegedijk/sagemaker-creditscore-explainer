# Deploying explainable models to AWS Sagemaker

Nowadays fairness and transparancy are becoming more and more important for 
machine learning applications. Especially when your machine learning model will
be used to make decisions with a large effect on people's wellbeing such as
acceptance or rejections of loan applications, fraud investigations, etc. 

Building some unscrutable black-box deep learning algorithms and claiming it has
an accuracy of over 90% is no longer good enough. Models and model predictions 
should be explainable to both regulators and end users.

However with modern approaches such as SHAP values (Lundberg et al 2017; 2018), 
the contributions of each feature to the final predictions can be calculated, 
thus providing an  explanation for how the model used the inputs to reach
its final prediction. The challenge is 
however to integrate such SHAP values into a production system, in order to 
provide transparent model decisions to stakeholders, decision-makers, regulators 
and customers alike.

Here we will be focusing on getting our model into production using Amazon Sagemaker.

It turns out that the `shap` library is not included by default in `sagemaker` 
estimator docker images, so it will not work out of the box.
Which means that in order to provide shap values as
part of your output, you have to build a custom docker container. Which is doable
but a little but more complicated than usual.

The other part that is different from standard models is constructing the response
payload to include shap values. But that should be the easy part. 

## notebook

It is easiest to deploy sagemaker containers, models and endpoints from within
a sagemaker notebook instance. So attach the git repository [https://github.com/oegedijk/sagemaker-creditscore-explainer](https://github.com/oegedijk/sagemaker-creditscore-explainer) to your sagemaker notebook instance.

More info on how to attach a git repo to your notebook instance here: [https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-git-repo.html](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-git-repo.html)

After you have connected to the repo, go to the `sagemaker/notebooks` directory
and open `sagemaker_explainer.ipynb`. 


## ECR Container

Given that we will be deploying a custom docker container, we need to make sure we 
have a ECR docker registry set up.

You can go the the AWS console, find `Elastic Container Registry`
and create one in your AWS region. For example, to create one in `eu-central-1` go 
to https://eu-central-1.console.aws.amazon.com/ecr/repositories?region=eu-central-1)

The name I gave to my repository is `sagemaker-explainer`.

#### setting ECR permissions

In order to create a custom docker container for our model we need to 
add `AmazonEC2ContainerRegistryFullAccess` policy to our notebook.

In the sagemaker console:

- click on notebook instances
- click on the notebook instance that you are using
- go to Permissions and encryption
- click on the `IAM role ARN`
- click on 'Attach Policies'
- find `AmazonEC2ContainerRegistryFullAccess`
- add it to the notebook.

#### Attach additional policies: 

You may have to add some additional permissions to your notebook policies, namely 
`"ecr:GetDownloadUrlForLayer"`, `"ecr:BatchGetImage"` and `"ecr:BatchCheckLayerAvailability"`. 

You can either edit these manually, or paste the following json:

```json
{
    "Version": "2008-10-17",
    "Statement": [
        {
            "Sid": "allowSageMakerToPull",
            "Effect": "Allow",
            "Action": [
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:BatchCheckLayerAvailability"
            ],
            "Resource": "*"
        }
    ]
}
```

#### Installing  `docker-credential-ecr-login`

In order to log into ECR from our sagemaker notebook, we need to install
a tool called `docker-credential-ecr-login`. We download and install it inside
out Sagemaker notebook with:

```
!sudo wget -P /usr/bin https://amazon-ecr-credential-helper-releases.s3.us-east-2.amazonaws.com/0.4.0/linux-amd64/docker-credential-ecr-login
```

and

```
!sudo chmod +x /usr/bin/docker-credential-ecr-login
```           

### Dockerfile

The `Dockerfile` to create our custom container is quite straightforward and
can be found in `sagemaker/container/Dockerfile`:

```Dockerfile
ARG SCIKIT_LEARN_IMAGE
FROM $SCIKIT_LEARN_IMAGE

COPY requirements.txt /requirements.txt
RUN pip install --no-cache -r /requirements.txt && \
    rm /requirements.txt
```

So basically we take a scikit learn image (defined by a parameter) and
then install additional requirements into it (basically `joblib` and `shap`).

#### DockerImage deployment helper class

In order to build and push our custom image we make use of this nice 
helper class:

```python
ecr_client = boto3.client("ecr", region_name=AWS_REGION)
docker_client = docker.APIClient()

class DockerImage:
    def __init__(self, registry, repository_name, tag="latest",
                docker_config_filepath='/home/ec2-user/.docker/config.json'):
        self.registry = registry
        self.repository_name = repository_name
        self.docker_config_filepath = docker_config_filepath
        self.tag = tag
        self._check_credential_manager()
        self._configure_credentials()

    def __str__(self):
        return "{}/{}:{}".format(self.registry, self.repository_name, self.tag)

    @property
    def repository(self):
        return "{}/{}".format(self.registry, self.repository_name)

    @property
    def short_name(self):
        return self.repository_name

    @staticmethod
    def _check_credential_manager():
        try:
            subprocess.run(
                ["docker-credential-ecr-login", "version"],
                stdout=subprocess.DEVNULL,
            )
        except Exception:
            raise Exception(
                "Couldn't run 'docker-credential-ecr-login'. "
                "Make sure it is installed and configured correctly."
            )

    def _configure_credentials(self):
        docker_config_filepath = Path(self.docker_config_filepath)
        if docker_config_filepath.exists():
            with open(docker_config_filepath, "r") as openfile:
                docker_config = json.load(openfile)
        else:
            docker_config = {}
        if "credHelpers" not in docker_config:
            docker_config["credHelpers"] = {}
        docker_config["credHelpers"][self.registry] = "ecr-login"
        docker_config_filepath.parent.mkdir(exist_ok=True, parents=True)
        with open(docker_config_filepath, "w") as openfile:
            json.dump(docker_config, openfile, indent=4)

    def build(self, dockerfile, buildargs):
        path = Path(dockerfile).parent
        for line in docker_client.build(
            path=str(path),
            buildargs=buildargs,
            tag=self.repository_name,
            decode=True,
        ):
            if "error" in line:
                raise Exception(line["error"])
            else:
                print(line)

    def push(self):
        docker_client.tag(
            self.repository_name, self.repository, self.tag, force=True
        )
        for line in docker_client.push(
            self.repository, self.tag, stream=True, decode=True
        ):
            print(line)
```

### Getting scikit-learn image URI

So now we can get our scikit-learn image:

```python
def scikit_learn_image():
    registry = sagemaker.fw_registry.registry(
        region_name=AWS_REGION, framework="scikit-learn"
    )
    repository_name = "sagemaker-scikit-learn"
    tag = "0.20.0-cpu-py3"
    return DockerImage(registry, repository_name, tag)

sklearn_image = scikit_learn_image()
```

### Building custom image based on scikit-learn image

And use that to build and push our custom image:

```python
def custom_image(aws_account_id, aws_region, repository_name, tag="latest"):
    ecr_registry = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com"
    return DockerImage(ecr_registry, repository_name, tag)

custom_image = custom_image(AWS_ACCOUNT_ID, AWS_REGION, ECR_REPOSITORY_NAME)

dockerfile = Path.cwd().parent / "container" / "Dockerfile"

custom_image.build(
    dockerfile=dockerfile,
    buildargs={'SCIKIT_LEARN_IMAGE': str(sklearn_image)}
)

custom_image.push()
```

This will take some time, but by the end you should have a custom training
image with shap installed. The URI will be along the lines of

'AWS_ACCOUNT_ID###.dkr.ecr.AWS_REGION###.amazonaws.com/sagemaker-explainer:latest'



### Training the Model

#### Sagemaker Estimator

Now that we have our custom training image, we can make use of the builtin 
Sagemaker `SKLearn` estimator, as long as we make sure to point it towards 
our custom image:

```python
estimator = SKLearn(
    image_name=str(custom_image),
    entry_point='entry_point.py',
    source_dir=str(source_dir),
    hyperparameters=hyperparameters,
    role=role,
    train_instance_count=1,
    train_instance_type='ml.m5.2xlarge', 
    output_path=output_path,
    code_location=output_path,
)
```

When training a custom model on sagemaker you have to define a training 
directory (`source_dir`) and a file that serves as the entry point 
(`entry_point`). This file must contain
the `train_fn` and the `model_fn`, `predict_fn`, `input_fn`, and `output_fn` 
for the inference later on. 

The training entry point can be found in `sagemaker/src/entry_point.py`:

```python
import os
import sys

# import training function
from model import parse_args, train_fn

# import deployment functions
from model import model_fn, predict_fn, input_fn, output_fn

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train_fn(args)
```

The training function itself is located in `sagemaker/src/model.py`. The main
difference with a regular model is that we also fit a `shap.TreeExplainer` to
the model and store this in our model directory:

```python

import argparse
import joblib
import os
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import shap

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier


class DFImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median", fill_value=None):
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        self.fitted = False
    
    def fit(self, X, y=None):
        self._feature_names = X.columns
        self.imputer.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X):
        assert self.fitted, "Need to cal .fit(X) function first!"
        return pd.DataFrame(
            self.imputer.transform(
                X[self._feature_names]), 
                columns=self._feature_names, 
                index=X.index
            ).astype(np.float32)
    
    def get_feature_names(self):
        return self._feature_names

class NumpyEncoder(json.JSONEncoder):
    """converts numpy arrays to lists before they get json encoded"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def parse_args(sys_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN_DATA"),
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST_DATA")
    )
    args, _ = parser.parse_known_args(sys_args)
    return args


def train_fn(args):
    print("loading data")
    train_df = pd.read_csv(args.train_data + "/train.csv", engine='python')
    test_df = pd.read_csv(args.test_data+ "/test.csv", engine='python')
    
    TARGET = 'SeriousDlqin2yrs'
    X_train = train_df.drop(TARGET, axis=1)
    y_train = train_df[TARGET]
    X_test = test_df.drop(TARGET, axis=1)
    y_test = test_df[TARGET]

    print("Imputing missing values")
    imputer = DFImputer(strategy='median').fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    print("Building model...")
    model = RandomForestClassifier(n_estimators=50, max_depth=6, max_leaf_nodes=30)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)

    print("Saving artifacts...")
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(imputer, open(str(model_dir / "imputer.joblib"), "wb"))
    joblib.dump(model, open(str(model_dir / "model.joblib"), "wb"))
    joblib.dump(explainer, open(str(model_dir / "explainer.joblib"), "wb"))

```

## Inference

For inference we have to define `model_fn` to load our model artifacts. Here
we have to make sure we also load the `explainer.joblib`. 

```python
def model_fn(model_dir):
    """loads artifacts from model_dir and bundle them in a model_assets dict"""
    model_dir = Path(model_dir)
    imputer= joblib.load(model_dir / "imputer.joblib")
    model = joblib.load(model_dir / "model.joblib")
    explainer = joblib.load(model_dir / "explainer.joblib")

    explainer = shap.TreeExplainer(model)

    model_assets = {
        "imputer": imputer,
        "model": model,
        "explainer": explainer
    }
    return model_assets
```

The function `input_fn` reads the JSON input and returns a dictionary with a 
pandas DataFrame. 

```python
def input_fn(request_body_str, request_content_type):
    """takes input json and returns a request dict with 'data' key"""
    assert request_content_type == "application/json", \
        "content_type must be 'application/json'"

    json_obj = json.loads(request_body_str)
    if isinstance(json_obj, str):
        # sometimes you have to unpack the json string twice for some reason. 
        json_obj = json.loads(json_obj)

    request = {
        'df': pd.DataFrame(json_obj)
    }
    return request
```

The `predict_fn` uses the input dataframe in `request` and the model assets
to calculate the predictions and shap values. 

We construct a return dictionary that includes both the prediction, 
the shap base value (comparable to an intercept in regular OLS), 
and the shap values per feature:

```python


def predict_fn(request, model_assets):
    """
    takes a request dict and model_assets dict and returns a response dict
    with 'prediction', 'shap_base' and 'shap_values'
    """ 
    print(f"data: {request['df']}")
    features = model_assets["imputer"].transform(request['df'])

    preds = model_assets["model"].predict_proba(features)[:, 1]

    expected_value = model_assets["explainer"].expected_value

    if expected_value.shape == (1,):
        expected_value = expected_value[0].tolist()
    else:
        expected_value = expected_value[1].tolist()

    shap_values = np.transpose(model_assets["explainer"].shap_values(features)[1])

    response = {}
    response['prediction'] = preds
    response['shap_base'] = expected_value
    response['shap_values'] = {k: v for k, v in zip(features.columns.tolist(), shap_values.tolist())}
    return response
```

And finally `output_fn` returns the response in JSON format:

```python
def output_fn(response, response_content_type):
    """takes a response dict and returns a json string of response"""
    assert (
        response_content_type == "application/json"
    ), "accept must be 'application/json'"
    response_body_str = json.dumps(response, cls=NumpyEncoder)
    return response_body_str
```

### Fitting the model

So now we simply fit our model as usual with:

```python
estimator.fit({'train_data': train_data, 'test_data': test_data})
```

### deploying the endpoint

And deploy it with:

```python
estimator.deploy(
    endpoint_name="credit-explainer",
    initial_instance_count=1, 
    instance_type='ml.c4.xlarge')
```

(this can take some time)

### Test the endpoint

We build a predictor with appropriate json serializers baked in and test the 
endpoint:

```python
from sagemaker.predictor import RealTimePredictor
from sagemaker.predictor import json_serializer, json_deserializer, CONTENT_TYPE_JSON

predictor = RealTimePredictor(
    endpoint=endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=json_serializer,
    deserializer=json_deserializer,
    content_type="application/json",
)

predictor.predict(test_df.sample(1).to_json(orient='records'))
```

Output should be something like:

```json
{'prediction': [0.020315580737022686],
 'shap_base': 0.06050333333333335,
 'shap_values': {'RevolvingUtilizationOfUnsecuredLines': [-0.018875482250995595],
  'age': [0.0026035737687252584],
  'NumberOfTime30-59DaysPastDueNotWorse': [-0.007295913630249845],
  'DebtRatio': [-0.001166559449290446],
  'MonthlyIncome': [0.00046746497026006246],
  'NumberOfOpenCreditLinesAndLoans': [-0.00012379074985687487],
  'NumberOfTimes90DaysLate': [-0.010730724822367846],
  'NumberRealEstateLoansOrLines': [-0.00029942272129598825],
  'NumberOfTime60-89DaysPastDueNotWorse': [-0.004091473195545041],
  'NumberOfDependents': [-0.0006754245156943097]}}
```

## lambda + api

Now all that is left to do is set up a lambda function to forward to API call to
your endpoint (Sagemaker endpoints can only be reached from within the AWS ecosystem, so you 
need to put a lambda function in between), and then get a public facing URL from
Gateway API. 

[This tutorial](https://medium.com/analytics-vidhya/invoke-an-amazon-sagemaker-endpoint-using-aws-lambda-83ff1a9f5443)
explains the various steps and configurations quite well, so just follow the steps.

In our case we simply forward the event payload onward without any repackaging,
so our `lambda_handler` is quite straightforward. I added a bunch of prints so 
that we can check our logs in case anything goes wrong. 

```python
import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                        ContentType='application/json',
                                        Body=event)

    print("raw response: ", response)
    result = json.loads(response['Body'].read().decode())
    print("decoded result: ", result)
    return result
```

## test api

After setting up the Gateway API you should have a public facing API url so we can
test our explainable model endpoint:

```python
# take a single sample row and convert it to JSON:
sample_json= df.sample(1)to_json(orient='records')

# define the header
header = {'Content-Type': 'application/json', 'Accept': 'application/json'}

# API url, copy your own here:
api_url = "https://#########.execute-api.eu-central-1.amazonaws.com/test/credit-explainer"


resp = requests.post(api_url, \
                    data=json.dumps(sample_json), \
                    headers=header)

print(resp.json())
```

And the result should again be something like this:

```json
{'prediction': [0.3045473967302875],
 'shap_base': 0.06050333333333335,
 'shap_values': {'RevolvingUtilizationOfUnsecuredLines': [0.017002446159576922],
  'age': [0.006427313815255611],
  'NumberOfTime30-59DaysPastDueNotWorse': [-0.007124554453726655],
  'DebtRatio': [-0.006844505153423333],
  'MonthlyIncome': [-0.019672520587649577],
  'NumberOfOpenCreditLinesAndLoans': [-0.010014011659840212],
  'NumberOfTimes90DaysLate': [0.288998818519516],
  'NumberRealEstateLoansOrLines': [-0.0007571802810589933],
  'NumberOfTime60-89DaysPastDueNotWorse': [-0.022098932417397806],
  'NumberOfDependents': [-0.001872810544298378]}}
```

In this case the customer was predicted to have 30% chance of a loan delinquency
in the next two years, mostly based on the number of times that they have been more 
than 90 days late in their repayments. 

Now you can put take this output and embed it in a dashboard where human
decision-makers or end customers have access to it!

Good luck building your own explainable machine learning models in AWS Sagemaker!




