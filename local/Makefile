all: help


help:           ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

env:		## create virtual environment and install dependencies
	python -m venv venv && \
	source venv/bin/activate && \
	pip install -r requirements.txt

model:	## build model artifacts and store them in /pkl/
	source venv/bin/activate && \
	python build_model_artifacts.py\

api:		## start the api and host it at http://localhost:5001/predict
	source venv/bin/activate && \
	gunicorn -b localhost:5001 api:flask_app

clean:		## remove pkl objects and venv
	rm pkl/*;\
	rm -r venv