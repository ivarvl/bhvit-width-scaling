# Binary ViT Width Scaling

Experiments on width scaling of binarized hybrid vision transformers. This repository
builds on **BHViT (Binarized Hybrid Vision Transformer)** and extends it with training,
fine-tuning, and profiling scripts to study how model width affects binary ViT accuracy
and efficiency.

This work was carried out as part of the
[Research Project 2026 Q4](https://cse3000-research-project.github.io/2026/Q4) of
[TU Delft](https://www.tudelft.nl/).

## Training and experiment scripts

All runnable scripts live in [`scripts/`](scripts/):

Model and sweep configurations are in [`configs/`](configs/).

Each run was repeated with seeds 0, 1, and 2, and reported results are averaged over
these three seeds.

## Acknowledgement

This work is based on **BHViT: Binarized Hybrid Vision Transformer**
by Tian Gao, Yu Zhang, Zhiyuan Zhang, Huajun Liu, Kaijie Yin, Chengzhong Xu, and Hui Kong
(CVPR 2025). Paper: https://arxiv.org/abs/2503.02394

```bibtex
@inproceedings{gao2025bhvit,
  title={BHViT: Binarized Hybrid Vision Transformer},
  author={Tian Gao and Zhiyuan Zhang and Yu Zhang and Huajun Liu and Kaijie Yin and Chengzhong Xu and Hui Kong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

BHViT in turn builds on [BinaryViT](https://github.com/Phuoc-Hoan-Le/BinaryViT) and
[DeiT](https://github.com/facebookresearch/deit).

## License

Released under the MIT License (see [LICENSE](LICENSE)).
