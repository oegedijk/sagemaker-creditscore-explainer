# Deploying explainable models to AWS Lambda using Zappa

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

Here we will be focusing on getting our model into production using Amazon Lambda 
functions and the awesome `zappa` package. The advantage of lambda functions
is that they are serverless: they don't need to be hosted on always-on infrastructure
like for example sagemaker endpoints (or your own EC2 deployment or on premise
solution). Only when they get called does amazon find a server to run the
specific code on. The downside is that there are a lot more restructions on
the size of the code and dependencies and infrastructure. (for example no GPUs 
sadly for deep learning deployments). All dependencies also have to be 
compatible with a particular flavor of Amazon Linux. 

Deploying and properly configuring an AWS lambda function and Gateway API (to
get a public face URL) can be quite tricky
and time consuming. Luckily there is an amazing almost magical package that simplifies all
of this for python projects called (Zappa)[https://github.com/Miserlou/Zappa].

Zappa takes care of all the configuration behind the scenes: bundling up all the code
and dependencies, setting up S3 buckets to store artifacts, 
setting up the lambda function, configuring the Gateway API, setting up logging, 
and just giving you nice API url in the end. 

For standard `scikit-learn` models zappa provides lambda compatible versions
of `numpy`, `pandas` and `scikit-learn`. However in order to deploy a model
that also returns `shap` values, we have to build our environment inside a docker
container as you will see.

# Zappa basics

The basics steps of deploying with zappa are:

1. Create a virtual environment
2. Install zappa into this venv along with other dependencies
3. Call `zappa init` or provide your own zappa_settings.json
4. call `zappa deploy`. 

Zappa automatically bundles up all packages in your venv in a zip file and sends
it to lambda to be deployed. If the package is too large for lambda limits (50MB zipped),
you can set `"slim_handler": true`, and zappa will only deploy a loader package to lambda,
store the actual package in an S3 bucket, and then download the package to lambda 
as needed in order to bypass the limit. 

## Zappa tutorial 
I highly recommend reading through this tutorial to find out what `zappa` does,
how to configure your permissions and deploy your first toy API to AWS lambda:
[https://pythonforundergradengineers.com/deploy-serverless-web-app-aws-lambda-zappa.html]
(https://pythonforundergradengineers.com/deploy-serverless-web-app-aws-lambda-zappa.html)

Especially setting up the right permissions can be quite challenging. 
After following this tutorial you should have a user set up with the right
credentials and saved the credentials to `~/.aws/credentials`. 

## Deploying a ML API including shap

When deploying to lambda you need to make sure that all packages are compatible 
with the Lambda AWS Linux environment. For standard packages such as `numpy`, 
`pandas` and `scikit-learn`, `zappa` already has pre-built packages prepared 
and will automatically substitute these for you. (told you it was magic!)


However `shap` is not one of those so you have to build compatible versions yourself.
The easiest way to do that is to install them inside a lambda compatible docker
container.


You can start from the `lambci/lambda:build-python3.7` image, and then run it in 
interactive mode. To have acces to your credentials you can mount the `~/.aws`
 directory to `/root/.aws`:

``docker run --rm -it -v $(pwd):/var/task -v ~/.aws:/root/.aws lambci/lambda:build-python3.7 /bin/bash``

Then you create the venv: `python -m venv lambda_env`, activate it:
`source lambda_env/bin/activate`, and install the dependencies `pip install -r requirements.txt`. 

The next step is to create a `zappa_settings.json` file by calling `zappa init`. 
The default options are fine. You then open the newly created `zappa_settings.json` 
and add the following lines:

```json
"slim_handler": true, // for large packages, store package in S3
"aws_region": "eu-west-1", // or whatever your aws region is
"keep_warm": false, // if you want to disable to default keep warm callback
```

If instead of calling `zappa init` you use the `zappa_settings.json` file in 
this repo, then also make sure to change the `"s3_bucket"` to something uniquely yours. 

Finally you call `zappa deploy dev` (assuming you named the task `dev`), and your api 
should get deployed. Like magic!

Note the url provided by GatewayAPI at the end, and go test your API!

## Using Makefile

In this project is also an example Makefile so that you can simply call 
`make env model deploy`to build and deploy the model, and then 
`make undeploy clean` to undeploy the project and clean up.

## Conclusion

Setting up lambda functins using zappa can be a breeze as long as you
build your environment in a proper lambda compatible environment.


## Appendix

### Makefile

Below the contents of the `Makefile`:

```Makefile
all: help

help:           ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

env:  			## build lambci compatible venv and install requirements.txt
	docker run --rm -v $(CURDIR):/var/task -v ~/.aws:/root/.aws lambci/lambda:build-python3.7 /bin/bash -c " \
	python -m venv lambda_env && \
	source lambda_env/bin/activate && \
	pip install -r requirements.txt && \
    zappa init"

model: 	## build the model artifacts: model.pkl, imputer.pkl, explainer.pkl
	docker run --rm -v $(CURDIR):/var/task -v ~/.aws:/root/.aws lambci/lambda:build-python3.7 /bin/bash -c " \
	source lambda_env/bin/activate && \
	python build_model_artifacts.py"

deploy:			## activate venv and deploy
	docker run --rm -v $(CURDIR):/var/task -v ~/.aws:/root/.aws lambci/lambda:build-python3.7 /bin/bash -c "\
	source lambda_env/bin/activate && \
	zappa deploy dev"

update:			## activate venv and update deployment
	docker run --rm -v $(CURDIR):/var/task -v ~/.aws:/root/.aws lambci/lambda:build-python3.7 /bin/bash -c "\
	source lambda_env/bin/activate && \
	zappa update dev --yes"

undeploy:		## activate venv and undeploy
	docker run --rm -v $(CURDIR):/var/task -v ~/.aws:/root/.aws lambci/lambda:build-python3.7 /bin/bash -c "\
	source lambda_env/bin/activate && \
	zappa undeploy dev --yes"

interactive:
	docker run --rm -it -v $(CURDIR):/var/task -v ~/.aws:/root/.aws lambci/lambda:build-python3.7 /bin/bash 

clean: 
	rm -r lambda_env
	rm pkl/*
```
