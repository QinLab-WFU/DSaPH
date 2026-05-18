# Deep Semantic-attention Proxy Hashing with Coarse-to-fine Representation for Multi-Label Remote Sensing Image Retrieval
This paper is accepted for publication in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

## Training
### Dependencies
We use python to build our code, you need to install the following packages to run：
- Python 3.10.0
- PyTorch 2.1.1
- torchvision 0.16.1
- CUDA 11.8
### Processing dataset
UCMerced: https://www.dropbox.com/s/u83ae1efaah2w9o/UCMercedLanduse.zip
MLRSNet:  https://data.mendeley.com/datasets/7j9bv9vwsx/3
DFC15:  https://drive.google.com/drive/folders/1TKGS6TIRxQ6a7gdaj0cHs-mRCtv_J1HA
### Start
After the dataset has been prepared, you could run the follow command to train.
> python main.py

## Citation
If you find this work useful for your research, please use the following.
```markdown
@article{yan2026deep,
  title={Deep Semantic-Attention Proxy Hashing With Coarse-to-Fine Representation for Multi-Label Remote Sensing Image Retrieval},
  author={Yan, Chunyu and Wang, Lei and Qin, Qibing and Huang, Lei and Zhang, Wenfeng},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2026},
  publisher={IEEE}
  }
