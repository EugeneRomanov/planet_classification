install_requirements:
	pip install -r requirements.txt
    
venv:
	python3 -m venv venv

download_dataset:
	gdown "https://drive.google.com/uc?id=1QirvafZfkTGf3-S47_z0cEpO4SIe0DiF" -O archive.zip
	unzip archive.zip -d ./dataset
	del archive.zip

train:
	python src/train.py configs/config.yaml

lint:
	flake8 src/
