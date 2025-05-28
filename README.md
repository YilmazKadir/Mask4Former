# Mask4Former: Mask Transformer for 4D Panoptic Segmentation (Renamed from MASK4D)
<div align="center">
<a href="https://github.com/YilmazKadir/">Kadir Yilmaz</a>, 
<a href="https://jonasschult.github.io/">Jonas Schult</a>,
<a href="https://nekrasov.dev/">Alexey Nekrasov</a>, 
<a href="https://www.vision.rwth-aachen.de/person/1/">Bastian Leibe</a>

RWTH Aachen University

Mask4Former is a transformer-based model for 4D Panoptic Segmentation, achieving a new state-of-the-art performance on the SemanticKITTI test set.

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg"></a>

![teaser](./docs/github_teaser.jpg)

</div>
<br><br>

[[Project Webpage](https://vision.rwth-aachen.de/Mask4Former)] [[arXiv](https://arxiv.org/abs/2309.16133)]

## News
* **2025-05-25**: MinkowskiEngine has been replaced with Spconv to simplify installation and enable 16-bit operations. Training now should be roughly twice as fast. See Minkowski in Tags for the previous version of the code.

* **2024-01-29**: Mask4Former accepted to ICRA 2024

* **2023-09-28**: Mask4Former on arXiv

### Dependencies
The main dependencies of the project are the following:
```yaml
python: 3.11
cuda: 12.4
```
We use [uv](https://docs.astral.sh/uv/getting-started/) as Python package and project manager. You can also set up your environment as you prefer and just use pip install command.

You can set up a uv virtual environment in your local directory as follows:
```
uv venv --python 3.11
source .venv/bin/activate
```

You can install the python packages as follows:
```
uv pip install -r requirements.txt --no-deps --extra-index-url https://download.pytorch.org/whl/cu124
```

### Data preprocessing
After installing the dependencies, we preprocess the SemanticKITTI dataset. This can take some time.

```
python -m datasets.preprocessing.semantic_kitti_preprocessing preprocess \
--data_dir $SEMANTICKITTI_DIR/SemanticKITTI/dataset \
--save_dir data/semantic_kitti \
--generate_instances True
```

### Training and testing
Train Mask4Former:
```bash
python main_panoptic_4d.py
```

In the simplest case the inference command looks as follows:
```bash
python main_panoptic_4d.py \
general.mode="validate" \
general.ckpt_path='PATH_TO_CHECKPOINT.ckpt'
```

Or you can use DBSCAN to boost the scores even further:
```bash
python main_panoptic_4d.py \
general.mode="validate" \
general.ckpt_path='PATH_TO_CHECKPOINT.ckpt' \
general.dbscan_eps=1.0
```
## Trained checkpoint
[Mask4Former](https://omnomnom.vision.rwth-aachen.de/data/mask4former/mask4former_spconv.ckpt)

The provided model, trained after the submission, achieves 71.3 LSTQ without DBSCAN and 72.0 with DBSCAN post-processing on the valiidation set.

## BibTeX
```
@inproceedings{yilmaz24mask4former,
  title     = {{Mask4Former: Mask Transformer for 4D Panoptic Segmentation}},
  author    = {Yilmaz, Kadir and Schult, Jonas and Nekrasov, Alexey and Leibe, Bastian},
  booktitle = {{International Conference on Robotics and Automation (ICRA)}},
  year      = {2024}
}
```
