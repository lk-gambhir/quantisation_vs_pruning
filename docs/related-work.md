# Related work and differentiation strategy

## Purpose

This is a focused starting map of closely related work found during a web search on 2026-10-29. It is **not an exhaustive systematic review** of every paper, notebook, or repository. Search results can miss work, and code repositories can change. Recheck links and publication details when writing the final report.

## Closest research papers

### 1. Tian, Luan, and Zheng — direct Fashion-MNIST overlap

“An Empirical Study on Model Pruning and Quantization,” BROADNETS 2022 proceedings (published 2023), [Springer DOI](https://doi.org/10.1007/978-3-031-40467-2_7). This is the closest prior work located. It compares pruning and quantization on AlexNet, ResNet18, VGG16BN, and VGG19BN across Fashion-MNIST, CIFAR-10, and UCI-HAR. Its abstract describes pruning/retraining, compression rate, architecture effects, and quantization/clustering choices. Therefore, this project must not claim to be the first comparison of pruning and quantization on Fashion-MNIST.

### 2. Kuzmin et al. — broader head-to-head benchmark

“Pruning vs Quantization: Which is Better?” NeurIPS 2023, [paper](https://arxiv.org/abs/2307.02973), [official research code](https://github.com/Qualcomm-AI-research/pruning-vs-quantization). The study analyzes pruning and quantization error and runs extensive experiments on large pretrained models/tasks. Its reported overall result is that quantization is better in most tested settings, while pruning can be competitive at high compression. The setting is much larger and more involved than a small Fashion-MNIST classroom study, but it establishes that the general question is not new.

### 3. Ganguli and Chong — pruning method on Fashion-MNIST

“Activation-Based Pruning of Neural Networks,” *Algorithms* (2024), [paper](https://www.mdpi.com/1999-4893/17/1/48), [GitHub code](https://github.com/tusharganguli/PrunedNetwork). This studies fully connected networks, including Fashion-MNIST, and compares activation-based pruning with magnitude pruning. It gives a direct reference for MLP pruning, although it does not make our deployment-benchmarking question redundant.

### 4. Han, Mao, and Dally — foundational compression pipeline

“Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding” (2015), [paper](https://arxiv.org/abs/1510.00149). The paper combines pruning, quantization, and Huffman coding. A student project can learn from the pipeline but should not imply that combining compression stages is new.

### 5. Schulze, Strassenburg, and Rabl — practical benchmark framing

“PQ Bench: Benchmarking Pruning and Quantization Techniques,” DEEM 2025, [paper](https://hpi.de/fileadmin/user_upload/fachgebiete/rabl/publications/2025/pq-bench.pdf), [code](https://github.com/hpides/pq-bench). It provides a more recent example of benchmarking multiple pruning granularities and quantization on actual execution setups. Consult it when designing measurements and interpreting hardware effects.

## Relevant code and notebooks

- [Qualcomm AI Research comparison code](https://github.com/Qualcomm-AI-research/pruning-vs-quantization): reference for the NeurIPS paper; heavy relative to this project and relies on large pretrained-model/data workflows.
- [PrunedNetwork](https://github.com/tusharganguli/PrunedNetwork): Fashion-MNIST and MNIST pruning methods, including magnitude pruning, in TensorFlow.
- [Deep Compression educational implementation](https://github.com/ciodar/deep-compression): PyTorch Lightning implementation of pruning, quantization, and Huffman coding, with MNIST and Imagenette examples.
- [Fashion-MNIST PyTorch Kaggle notebook](https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy/notebook): a classifier baseline, not the exact compression comparison.
- Kaggle has Fashion-MNIST competition notebooks and data, but the search did not identify a clearly documented, high-quality Kaggle notebook that matches this project's full pruning/quantization/size/latency protocol. Use Kaggle notebooks as implementation examples only, and verify their evaluation details before comparing results.
- [Official Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist): dataset description, benchmark results, and citation. Treat its varied model scores as context, not directly comparable results, because models and preprocessing differ.

## What can honestly distinguish this classroom project

The contribution should be described as **a controlled, reproducible case study**, not a new compression method. The project can be distinct in its experimental question and evidence:

1. **Compare at a meaningful budget.** Do not compare “50% pruned” with “INT8” as if the levels were automatically equivalent. Report complete trade-off curves and identify the best method under a predeclared accuracy-loss limit (for example, within 1 percentage point of baseline) and/or a file-size budget.
2. **Measure deployment outcomes separately.** Report zero-weight sparsity, actual serialized artifact bytes, and measured CPU latency as three distinct quantities. Unstructured pruning may increase zero count without shrinking the dense file or making dense CPU execution faster. Making this visible is a strong, honest engineering result.
3. **Use paired variants and fixed evaluation.** Every compressed variant should derive from the same trained baseline for each seed; test samples and evaluator are held constant; use three seeds if feasible; report spread and paired changes.
4. **Add one focused pruning granularity comparison.** If time allows, compare unstructured weight pruning to structured neuron pruning (for the MLP) or channel/filter pruning (if a CNN is added). Structured pruning physically reduces matrix/channel dimensions and provides a more plausible dense-kernel speed benefit. This is more informative than adding many unrelated compression methods.
5. **Optionally add a retraining ablation.** Compare one-shot pruning with a fixed prune-then-fine-tune recipe at one or two sparsity levels. Keep this secondary so the core project remains manageable. Clearly label the extra training as a distinct intervention.
6. **Make the pipeline reusable.** Publish configuration files, environment versions, raw timing data, saved metrics, plotting code, and exact run instructions. This improves reproducibility but does not by itself make the research question novel.

## Recommended final framing

> “Prior papers have compared pruning and quantization, including on Fashion-MNIST. This project reproduces the comparison on a small, fixed classifier and focuses on a practical question: at a stated accuracy budget, which method changes actual model file size and measured CPU latency, and does unstructured sparsity translate into deployment gains? Results are repeated across seeds and reported with the exact implementation and hardware.”

That framing is defensible even if the result matches prior literature. A replication with careful measurement is more credible than claiming originality that a teacher can disprove with a quick search.
