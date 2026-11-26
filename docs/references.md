# References and implementation resources

These sources support the methods used in this study. They are starting references, not substitutes for reading and citing the exact methods used in the final report.

## Research papers

- Andrey Kuzmin, Markus Nagel, Mart van Baalen, Arash Behboodi, and Tijmen Blankevoort. “Pruning vs Quantization: Which is Better?” 2023. [arXiv:2307.02973](https://arxiv.org/abs/2307.02973). Direct comparison and motivation for measuring the methods separately under controlled conditions.
- Yuzhe Tian, Tom H. Luan, and Xi Zheng. “An Empirical Study on Model Pruning and Quantization.” BROADNETS 2022 proceedings, published 2023. [Springer DOI](https://doi.org/10.1007/978-3-031-40467-2_7). Directly relevant: evaluates AlexNet, ResNet18, VGG16BN, and VGG19BN on Fashion-MNIST, CIFAR-10, and UCI-HAR.
- Tushar Ganguli and Edwin K. P. Chong. “Activation-Based Pruning of Neural Networks.” *Algorithms*, 17(1), 48, 2024. [Paper](https://www.mdpi.com/1999-4893/17/1/48), [code](https://github.com/tusharganguli/PrunedNetwork). Relevant to fully connected networks and Fashion-MNIST; compares activation-based with magnitude pruning.
- Song Han, Huizi Mao, and William J. Dally. “Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.” 2015. [arXiv:1510.00149](https://arxiv.org/abs/1510.00149). Foundational work on combining network compression techniques.
- Raghuraman Krishnamoorthi. “Quantizing deep convolutional networks for efficient inference: A whitepaper.” 2018. [arXiv:1806.08342](https://arxiv.org/abs/1806.08342). Background on practical quantization choices.
- Jonas Schulze, Nils Strassenburg, and Tilmann Rabl. “PQ Bench: Benchmarking Pruning and Quantization Techniques.” DEEM 2025. [Paper](https://hpi.de/fileadmin/user_upload/fachgebiete/rabl/publications/2025/pq-bench.pdf), [code](https://github.com/hpides/pq-bench). Useful for designing comparisons that include pruning granularity and actual benchmark measurements.

## Official implementation/data documentation

- PyTorch, [Quantization Recipe](https://docs.pytorch.org/tutorials/recipes/quantization.html): supported workflows and limitations, including dynamic quantization support for `Linear` and `LSTM` modules in the described path.
- PyTorch, [Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html): pruning APIs, serialization, and making pruning permanent.
- PyTorch, [Fashion-MNIST loading example](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html): dataset characteristics and TorchVision loading.
- TorchVision, [FashionMNIST dataset API](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html): dataset interface.
- Qualcomm AI Research, [pruning-vs-quantization repository](https://github.com/Qualcomm-AI-research/pruning-vs-quantization): implementation accompanying Kuzmin et al.; intended for larger pretrained models and tasks, not a drop-in Fashion-MNIST student pipeline.
- Ciodar, [deep-compression repository](https://github.com/ciodar/deep-compression): educational PyTorch Lightning implementation of pruning, quantization, and Huffman coding, with MNIST and Imagenette examples.
- [Kaggle Fashion-MNIST PyTorch classifier notebook](https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy/notebook): useful only as a dataset/model-training baseline, not as evidence for compression comparisons.
- Kaggle's [Fashion-MNIST competition page](https://www.kaggle.com/c/fashion-mnist-competition/data): an additional source of ordinary classification baselines; competition data and split format differ from the original TorchVision benchmark, so do not mix its scores with this study.

## Citation practice for final report

- Cite a paper for the method motivation and primary claims; cite official docs for the exact API/workflow used.
- Check publication metadata and any changed framework APIs during implementation.
- Do not cite this roadmap as evidence of experimental results.
- The broad pruning-versus-quantization question and Fashion-MNIST applications already have prior work; describe this as a controlled replication/application study and cite that work rather than claiming the comparison is new.
