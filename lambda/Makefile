all: help

help:           ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

env:  			## build lambci compatible venv and install requirements.txt
	docker run --rm -v $(CURDIR):/var/task -v ~/.aws:/root/.aws lambci/lambda:build-python3.7 /bin/bash -c " \
	python -m venv lambda_env && \
	source lambda_env/bin/activate && \
	pip install -r requirements.txt"

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

interactive:	## start interactive session inside lambci docker container
	docker run --rm -it -v $(CURDIR):/var/task -v ~/.aws:/root/.aws lambci/lambda:build-python3.7 /bin/bash 

clean: 			## remove lambda_env en pkl/*
	rm -r lambda_env
	rm pkl/*


