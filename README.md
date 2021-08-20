![Python](https://img.shields.io/badge/python-3.8-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.2.3-green?style=flat-square&logo=tensorflow)

# On the Importance of Encrypting Deep Features

## Overview

In this study, we analyze model inversion attacks with only two assumptions: feature vectors of user data are known, and a black-box API for inference is provided.
On the one hand, limitations of existing studies are addressed by opting for a more practical setting.
Experiments have been conducted on state-of-the-art models in person re-identification, and two attack scenarios (i.e., recognizing auxiliary attributes and reconstructing user data) are investigated.
Results show that an adversary could successfully infer sensitive information even under severe constraints.
On the other hand, it is advisable to encrypt feature vectors, especially for a machine learning model in production.
As an alternative to traditional encryption methods such as AES, a simple yet effective method termed ShuffleBits is presented.
More specifically, the binary sequence of each floating-point number gets shuffled.
Deployed using the one-time pad scheme, it serves as a plug-and-play module that is applicable to any neural network, and the resulting model directly outputs deep features in encrypted form.
Source code is publicly available at https://github.com/nixingyang/ShuffleBits.

## Environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda config --set auto_activate_base false
conda create --yes --name TensorFlow2.2 python=3.8
conda activate TensorFlow2.2
conda install --yes cudatoolkit=10.1 cudnn=7.6 -c nvidia
conda install --yes matplotlib numpy=1.18 pandas pydot scikit-learn
pip install tensorflow==2.2.3 tensorflow-addons==0.11.2
pip install opencv-python
```

## Extract Feature Vectors

Follow the instructions in [FastReID v1.0.0](https://github.com/JDAI-CV/fast-reid/tree/v1.0.0).
Extract feature vectors of samples in the Market-1501 and DukeMTMC-reID datasets.
The following pre-trained models are utilized:
| config-file | MODEL.WEIGHTS |
| - | - |
| ./configs/MSMT17/bagtricks_R50.yml | ./model_zoo/msmt_bot_R50.pth |
| ./configs/MSMT17/AGW_R50.yml | ./model_zoo/msmt_agw_R50.pth |
| ./configs/MSMT17/sbs_R50.yml | ./model_zoo/msmt_sbs_R50.pth |

Save feature vectors to disk. You may replace [this snippet](https://github.com/JDAI-CV/fast-reid/blob/v1.0.0/fastreid/engine/defaults.py#L468-L469) with
```python
model_name = os.path.basename(cfg.MODEL.WEIGHTS).split(".")[0]
if os.path.isdir(dataset_name):
    output_file_path = os.path.join(dataset_name, "{}.npz".format(model_name))
else:
    output_file_path = os.path.join(
        cfg.OUTPUT_DIR, "{}_{}.npz".format(model_name, dataset_name))
if os.path.isfile(output_file_path):
    continue
logger.info("Saving to {}...".format(output_file_path))
accumulated_image_file_paths, accumulated_feature_array = inference_on_dataset(
    model, data_loader, evaluator, flip_test=cfg.TEST.FLIP_ENABLED)
np.savez(output_file_path,
         accumulated_image_file_paths=accumulated_image_file_paths,
         accumulated_feature_array=accumulated_feature_array)
```

## Recognizing Auxiliary Attributes and Reconstructing User Data

```bash
python3 -u scripts/reconstruction_attributes.py --dataset_name "?" --feature_file_path "?"
```
```bash
python3 -u scripts/reconstruction_autoencoder.py --dataset_name "?" --feature_file_path "?" --pixel_loss_weight 1.0 --feature_reconstruction_loss_weight 0.0
```
```bash
python3 -u scripts/reconstruction_GAN.py --dataset_name "?" --feature_file_path "?" --pixel_loss_weight 0.0 --feature_reconstruction_loss_weight 1.0 --discriminator_loss_weight 0.06
```

- The `dataset_name` argument can be `"Market1501"` or `"DukeMTMC_reID"`.
- The `feature_file_path` argument refers to the path of the corresponding `npz` file.
- The `*_loss_weight` arguments define the weights of different loss functions.

## Evaluation of the Reconstructed Images

```bash
python3 -u scripts/evaluation.py --source_feature_file_path "?" --target_feature_folder_path "?"
```

- The `source_feature_file_path` argument points to the `npz` file which contains the feature vectors of the ground truth images.
- The `target_feature_folder_path` argument points to the folder where the `npz` files for the reconstructed images are stored.

## ShuffleBits

```bash
python3 -u scripts/encryption_utils.py
```

## Citation

Please consider citing [this work](https://arxiv.org/abs/2108.07147) if it helps your research.

```
@misc{ni2021importance,
      title={On the Importance of Encrypting Deep Features}, 
      author={Xingyang Ni and Heikki Huttunen and Esa Rahtu},
      year={2021},
      eprint={2108.07147},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```