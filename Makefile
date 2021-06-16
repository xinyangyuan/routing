.PHONY: data-wget data-cli data-2000 tensorboard jupyter-notebook jupyter-lab run score

ifneq ($(shell which rc-cli), )
data: data-cli
# else ifneq ($(shell where rc-cli), )
# data: data-cli
else
data: data-wget
endif

data-wget:
	wget https://cave-competition-app-data.s3.amazonaws.com/amzn_2021/data.tar.xz
	tar -vxf data.tar.xz

data-cli:
	rc-cli reset-data

data-2000:
	wget -O data/model_build_inputs/route_data.json https://raw.githubusercontent.com/JGIoA/temp/main/route_data_high_2000.json  

tb: tensorboard

tensorboard:
	tensorboard --logdir=experiments --port=8888 --bind_all

jt: jupyter-notebook

jupyter-notebook:
	jupyter notebook --ip=0.0.0.0 --port=7777 --NotebookApp.token='xyj1!' --no-browser &

jl: jupyter-lab

jupyter-lab:
	jupyter lab --ip=0.0.0.0 --port=7777 --NotebookApp.token='xyj1!' --no-browser &

run:
	python  src/train.py --model_dir experiments/base_model

score:
	python src/score.py