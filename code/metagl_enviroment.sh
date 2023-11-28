#!/bin/bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy==1.21.5
pip install scikit-learn==1.0.2
pip install pandas
pip install networkx
pip install tqdm
pip install psutil
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install transformers

