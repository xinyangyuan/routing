.PHONY: data-wget data-cli

ifneq ($(shell which rc-cli), )
data: data-cli
else ifneq ($(shell where rc-cli), )
data: data-cli
else
data: data-cli
endif

data-wget:
	wget https://cave-competition-app-data.s3.amazonaws.com/amzn_2021/data.tar.xz
	tar -vxf data.tar.xz

data-cli:
	rc-cli reset-data