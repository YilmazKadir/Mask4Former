# MASK4D: Mask Transformer for 4D Panoptic Segmentation 
<div align="center">
<a href="https://github.com/YilmazKadir/">Kadir Yilmaz</a>, 
<a href="https://jonasschult.github.io/">Jonas Schult</a>,
<a href="https://nekrasov.dev/">Alexey Nekrasov</a>, 
<a href="https://www.vision.rwth-aachen.de/person/1/">Bastian Leibe</a>

RWTH Aachen University

MASK4D is a transformer-based model for 4D Panoptic Segmentation, achieving a new state-of-the-art performance on the SemanticKITTI test set.

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg"></a>

![teaser](./docs/github_teaser.jpg)

</div>
<br><br>

[[Project Webpage](https://vision.rwth-aachen.de/mask4d)] [[arXiv](https://arxiv.org/abs/2309.16133)]

## News

* **2023-09-28**: Paper on arXiv

### Dependencies
The main dependencies of the project are the following:
```yaml
python: 3.8
cuda: 11.7
```
You can set up a conda environment as follows
```
conda create --name mask4d python=3.8
conda activate mask4d
pip install -r requirements.txt

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

cd third_party/pointnet2 && python setup.py install

cd ..
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install
cd ../..
```

### Data preprocessing
After installing the dependencies, we preprocess the SemanticKITTI dataset.

```
python -m datasets.preprocessing.semantic_kitti_preprocessing preprocess \
--data_dir "PATH_TO_RAW_SEMKITTI_DATASET" \
--save_dir "data/semantic_kitti"

python -m datasets.preprocessing.semantic_kitti_preprocessing make_instance_database \
--data_dir "PATH_TO_RAW_SEMKITTI_DATASET" \
--save_dir "data/semantic_kitti"
```

### Training and testing
Train MASK4D:
```bash
python main_panoptic.py
```

In the simplest case the inference command looks as follows:
```bash
python main_panoptic.py \
general.mode="validate" \
general.ckpt_path='PATH_TO_CHECKPOINT.ckpt'
```

Or you can use DBSCAN to boost the scores even further:
```bash
python main_panoptic.py \
general.mode="validate" \
general.ckpt_path='PATH_TO_CHECKPOINT.ckpt' \
general.dbscan_eps=1.0
```
## Trained checkpoint
[MASK4D](https://omnomnom.vision.rwth-aachen.de/data/mask4d/mask4d.ckpt)

The provided model, trained after the submission, achieves 71.1 LSTQ without DBSCAN and 71.5 with DBSCAN post-processing.

## BibTeX
```
@article{yilmaz2023mask4d,
  title     = {{MASK4D: Mask Transformer for 4D Panoptic Segmentation}},
  author    = {Yilmaz, Kadir and Schult, Jonas and Nekrasov, Alexey and Leibe, Bastian},
  journal   = {arXiv prepring arXiv:2309.16133},
  year      = {2023}
}
```
