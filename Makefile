# U-Det 常用命令（AutoDL 环境）
# 说明：make 无法激活 conda 环境，因此这里直接使用环境内的 python 绝对路径。
# 如环境路径不同，可覆盖：make PY=/path/to/python test

PY ?= /root/miniconda3/envs/udet/bin/python

.PHONY: help env data prepare encoder test smoke clean-processed all

help:
	@echo "可用命令："
	@echo "  make env              创建 conda 环境并安装依赖（Python 3.12）"
	@echo "  make data             下载 CoDET-M4 全量数据（hf-mirror）"
	@echo "  make prepare          数据统计 + 划分 train/val/test"
	@echo "  make encoder          下载 CodeT5-base 权重"
	@echo "  make test             运行单元测试"
	@echo "  make smoke            端到端冒烟测试（迷你权重）"
	@echo "  make all              一键：数据 + 权重 + 准备 + 测试"
	@echo "  make clean-processed  清空 data/processed"

env:
	conda create -n udet python=3.12 -y
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

prepare:
	$(PY) scripts/prepare_data.py

encoder:
	$(PY) scripts/download_encoder.py

test:
	$(PY) -m pytest -q

smoke:
	$(PY) scripts/smoke_test.py

all: data encoder prepare test smoke

clean-processed:
	rm -rf data/processed/*
