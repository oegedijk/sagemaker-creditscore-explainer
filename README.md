# README

by: Oege Dijk

This repository is a demonstration of how to deploy explainable machine learning
models (using SHAP) to AWS cloud infrastructure. 

Three approaches are shown:

1. Using Sagemaker. This involves setting up custom training containers on ECR,
and defining the proper inference functions.
    - All notebooks, code, configurations, Dockerfiles and READMEs can be found
    in the [sagemaker](https://github.com/oegedijk/sagemaker-creditscore-explainer/tree/master/sagemaker) folder.
2. Using straight AWS Lambda functions and zappa. This is easier, but still a 
number of tricky things to get to work (such as deploying from inside a lambda 
compatible docker container)
    - All code, Makefiles and READMEs can be found in the 
    [lambda](https://github.com/oegedijk/sagemaker-creditscore-explainer/tree/master/lambda) folder.
3. For completeness an example of a local on premise deployment can be in the
    [local](https://github.com/oegedijk/sagemaker-creditscore-explainer/tree/master/local) folder.


An example dashboard that sends requests to both a sagemaker deployment and a lambda
deployment is running at [http://creditexplainer.herokuapp.com](http://creditexplainer.herokuapp.com)

