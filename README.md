<div align="center">

# EgoExoMem

### Cross-View Memory Reasoning over Synchronized Egocentric and Exocentric Videos

[![Paper](https://img.shields.io/badge/arXiv-2605.18734-b31b1b.svg)](https://arxiv.org/pdf/2605.18734)
[![Data](https://img.shields.io/badge/Data-GitHub-blue.svg)](https://github.com/RuipingL/EgoExoMem/tree/main/data)
[![Code](https://img.shields.io/badge/Code-rag__baselines-orange.svg)](https://github.com/RuipingL/EgoExoMem/tree/main/code/rag_baselines)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](#license)

<p align="center">
  <img src="assets/teaser.png" width="95%" alt="EgoExoMem teaser"/>
</p>

</div>

## Overview

Humans naturally encode experiences from two complementary perspectives: field memory, which relives events through a first-person viewpoint, and observer memory, which recalls the same event from a third-person vantage point. These perspectives mirror the brain's parallel egocentric and allocentric reference frames, and their interplay supports situation awareness. Yet existing ego-exo works are largely motivated by imitation learning and either treat the two streams independently or evaluate only local view-clip matching, rather than reasoning over the full, synergistic memory formed by both viewpoints.

EgoExoMem is the first benchmark for memory-based reasoning over synchronized egocentric and exocentric video. It reveals that neither view alone suffices for comprehensive understanding, and that existing MLLMs and memory mechanisms fail to fully exploit dual-view complementarity. Alongside the benchmark we propose **E²-Select**, a training-free frame selection method that achieves superior performance via relevance-based budget allocation and Cholesky-based k-DPP sampling that account for view asymmetry and cross-view temporal consistency.

## Benchmark

EgoExoMem comprises 2,665+ human-verified multiple-choice QA pairs drawn from EgoExo4D and LEMMA. Memory is characterized along eight QA types spanning spatial reasoning, temporal reasoning, view dependency, and memory time span.

<p align="center">
  <img src="assets/qa_types.png" width="95%" alt="EgoExoMem QA types"/>
</p>

| ID | QA Type | Focus |
|---|---|---|
| Q1 | Habitual Location | initial location of an object |
| Q2 | Instantaneous Position | position at a referenced moment |
| Q3 | Resulting Location | where an object ends up after an action |
| Q4 | Egocentric Direction | direction relative to the wearer |
| Q5 | Object State | state of an object at a referenced moment |
| Q6 | Allocentric Relation | object-to-object spatial relation |
| Q7 | Third Person Activity | what another person is doing and where |
| Q8 | Temporal Ordering | order of a sequence of activities |

## Method

E²-Select addresses multi-view frame selection where existing methods, designed for single-stream video, do not transfer. It explicitly accounts for view dependency, temporal consistency, and cross-view synchronization through two components: relevance-based budget allocation that distributes the frame budget across the ego and exo streams by query-frame relevance, and Cholesky-based k-DPP sampling that selects a diverse, non-redundant frame subset within each view.

## Key Findings

1. Neither view alone suffices. Retrieval from both views consistently outperforms single-view retrieval.
2. The egocentric view dominates over the exocentric view overall, but the two are complementary across QA types.
3. Frame selection strategies outperform RAG-based methods on the benchmark.
4. Failure analysis exposes a systematic view-dependency mismatch for Third Person Activity, motivating query-aware view routing as future work.

## Data and Code

The benchmark data is available at [`data/`](https://github.com/RuipingL/EgoExoMem/tree/main/data), and the RAG and frame-selection baselines are available at [`code/rag_baselines/`](https://github.com/RuipingL/EgoExoMem/tree/main/code/rag_baselines).

```bash
git clone https://github.com/RuipingL/EgoExoMem.git
cd EgoExoMem
```

## Citation

```bibtex
@article{egoexomem,
  title   = {EgoExoMem: Cross-View Memory Reasoning over Synchronized Egocentric and Exocentric Videos},
  author  = {Liu, Ruiping and others},
  journal = {arXiv preprint arXiv:2605.18734},
  year    = {2026}
}
```

## License

Released under the CC BY 4.0 license. Source videos follow the original terms of EgoExo4D and LEMMA.

## Acknowledgements

This work was conducted at the Computer Vision for Human-Computer Interaction Lab (CV:HCI), Karlsruhe Institute of Technology.
