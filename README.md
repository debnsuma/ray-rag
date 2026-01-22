# Building Scalable RAG for Agentic AI with Ray

[![Ray](https://img.shields.io/badge/Ray-2.53+-blue)](https://docs.ray.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)](https://docs.trychroma.com/)

**Session Materials for [Agentic AI Summit 2026](https://www.summit.ai/)**
**Date:** January 22, 2026

---

## Overview

This repository contains hands-on workshop materials demonstrating how to build production-grade Retrieval-Augmented Generation (RAG) pipelines using Ray's distributed computing framework. The content progresses from Ray fundamentals to a complete end-to-end RAG system capable of scaling across distributed clusters.

RAG systems are foundational to modern agentic AI applications, enabling LLMs to ground their responses in retrieved context. By leveraging Ray, we can scale these systems from prototype to production while maintaining resource efficiency and fault tolerance.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Module Overview](#module-overview)
  - [Module 0: Ray Core Introduction](#module-0-ray-core-introduction)
  - [Module 1: Ray Data Processing](#module-1-ray-data-processing)
  - [Module 2: Embeddings Generation and Retrieval](#module-2-embeddings-generation-and-retrieval)
  - [Module 3: LLM Processing with Ray Actors](#module-3-llm-processing-with-ray-actors)
  - [Module 4: End-to-End RAG Pipeline](#module-4-end-to-end-rag-pipeline)
- [Architecture](#architecture)
- [Key Concepts](#key-concepts)
- [Getting Started](#getting-started)
- [Resources](#resources)

---

## Prerequisites

- Python 3.10+
- Basic familiarity with distributed systems concepts
- Understanding of embeddings and vector databases
- Experience with PyTorch/Transformers (helpful but not required)

**Required Libraries:**
```
ray>=2.53.0
sentence-transformers
chromadb
transformers
numpy
pandas
pyarrow
```

---

## Repository Structure

```
ray-rag/
├── Module_00_Ray_Core_intro.ipynb             # Ray fundamentals
├── Module_01_Ray_Data.ipynb                   # Distributed data processing
├── Module_02_Embeddings_Gen_Retrieval.ipynb   # Embeddings and vector DB
├── Module_03_LLM_Ray_Actor_Processing.ipynb   # LLM inference with actors
├── Module_04_RAG_Pipelines_Enhanced.ipynb     # Complete RAG pipeline (enhanced)
├── extra/code/                                # Supporting Python scripts
│   ├── counter.py                             # Actor state management demo
│   ├── memory_inspection.py                   # Memory optimization patterns
│   ├── parallel_process.py                    # Task parallelization example
│   ├── ray_actor.py                           # Actor pattern examples
│   └── sequential_process.py                  # Sequential baseline
├── assets/                                    # Diagrams and images
├── around.txt                                 # Sample corpus (Around the World in 80 Days)
└── prompts.parquet                            # Pre-generated query dataset
```

---

## Module Overview

### Module 0: Ray Core Introduction

Introduces the foundational primitives of Ray's distributed execution model.

**Topics Covered:**
- The `@ray.remote` decorator for converting functions and classes
- **Tasks**: Stateless remote function execution for embarrassingly parallel workloads
- **Actors**: Stateful remote class instances for maintaining state across invocations
- **ObjectRef**: Futures representing pending computation results
- `ray.get()` and `ray.put()` for data retrieval and shared memory optimization

**Key Demonstration:**
Processing 8 images sequentially (8 seconds) vs. parallel execution (~1 second) on an 8-core machine.

---

### Module 1: Ray Data Processing

Covers Ray Datasets for scalable, distributed data processing pipelines.

**Topics Covered:**
- Reading data from various sources (S3, local filesystem, Parquet)
- Lazy execution model and streaming computation
- Transformation operations: `map_batches()`, `repartition()`, `random_shuffle()`, `sort()`, `groupby()`
- Stateless (task-based) vs. stateful (actor-based) batch processing
- Actor pool strategies for compute-intensive transformations
- Resource allocation and batch size tuning

**Key Pattern:**
Using stateful actors with `map_batches()` for operations requiring expensive initialization (model loading, database connections).

---

### Module 2: Embeddings Generation and Retrieval

Implements the embedding generation and vector storage components of a RAG system.

**Topics Covered:**
- Generating embeddings using SentenceTransformers (`hkunlp/instructor-large`)
- 768-dimensional dense vector representations
- ChromaDB integration for vector storage and retrieval
- Persistent vs. in-memory vector stores
- Semantic similarity search with configurable top-k retrieval
- Actor-based wrappers for concurrent database access

**Key Components:**
- `DocEmbedder`: Stateful actor for batch embedding generation (GPU-accelerated)
- `ChromaWrapper`: Actor managing concurrent read/write access to vector store

---

### Module 3: LLM Processing with Ray Actors

Demonstrates efficient LLM inference patterns using Ray's actor model.

**Topics Covered:**
- Integrating Hugging Face Transformers with Ray
- Task-based vs. actor-based LLM inference (performance comparison)
- Model state persistence across inference calls
- Fractional GPU allocation for cost-efficient scaling
- Batch inference patterns for throughput optimization

**Key Insight:**
Task-based inference reloads the model on every call (inefficient). Actor-based inference maintains the model in GPU memory, amortizing initialization cost across many requests.

**Model Used:** `Qwen/Qwen2.5-0.5B-Instruct`

---

### Module 4: End-to-End RAG Pipeline

Assembles all components into a production-grade RAG pipeline with comprehensive documentation and architecture diagrams.

**Pipeline Stages:**
1. **Document Ingestion** - Load and process text corpus
2. **Embedding Generation** - Convert documents to vector representations
3. **Vector Storage** - Persist embeddings to ChromaDB
4. **Query Processing** - Load prompts and generate query embeddings
5. **Document Retrieval** - Semantic search against vector store
6. **Context Injection** - Augment prompts with retrieved documents
7. **LLM Generation** - Generate responses using retrieved context
8. **Output Persistence** - Store results to Parquet

**Key Pattern:**
Chaining multiple `map_batches()` operations with different actor pools, each with appropriate resource allocations (GPU fractions, batch sizes, concurrency levels).

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline Architecture                          │
└────────────────────────────────────────────────────────────────────────────┘

    Input Queries (Parquet)
           │
           ▼
    ┌────────────────────┐
    │   Embedder Pool    │  ← SentenceTransformer, GPU 0.1/actor
    │   (Actor-based)    │
    └─────────┬──────────┘
              │ queries + embeddings
              ▼
    ┌────────────────────┐
    │  ChromaDB Reader   │  ← Vector similarity search
    │   (Actor-based)    │
    └─────────┬──────────┘
              │ queries + embeddings + retrieved docs
              ▼
    ┌────────────────────┐
    │  Prompt Enhancer   │  ← Context injection
    │   (Actor-based)    │
    └─────────┬──────────┘
              │ enhanced prompts
              ▼
    ┌────────────────────┐
    │    Chat Pool       │  ← LLM inference, GPU 0.15/actor
    │   (Actor-based)    │
    └─────────┬──────────┘
              │ generated responses
              ▼
    Output Results (Parquet)
```

---

## Key Concepts

### Tasks vs. Actors

| Aspect         | Tasks                              | Actors                                  |
|----------------|------------------------------------|-----------------------------------------|
| State          | Stateless                          | Stateful                                |
| Use Case       | Independent parallel operations    | Operations requiring shared state       |
| Initialization | Per-invocation                     | Once at creation                        |
| Best For       | Data transformation, map operations| Model serving, connection pools         |

### The Callable Class Pattern

```python
class DocEmbedder:
    def __init__(self):
        # Expensive setup - runs ONCE per actor
        self._model = SentenceTransformer("model-name")

    def __call__(self, batch):
        # Batch processing - runs MANY times
        return self._model.encode(batch["text"])
```

### Resource Allocation Patterns

```python
# Fractional GPU allocation for efficient multi-tenancy
.map_batches(
    Embedder,
    compute=ray.data.ActorPoolStrategy(size=4),
    num_gpus=0.1,      # 10% of a GPU per actor
    batch_size=8       # Records per batch
)
```

### Memory Efficiency

Ray's object store enables zero-copy data sharing between tasks:
- Large arrays are stored in shared memory
- Workers receive references, not copies
- Verified via `array.flags.owndata == False`

---

## Getting Started

1. **Initialize Ray:**
```python
import ray
ray.init()
```

2. **Verify cluster resources:**
```python
print(ray.cluster_resources())
```

3. **Work through modules sequentially:**
   - Start with Module 0 to understand Ray primitives
   - Progress through Module 4 for the complete RAG implementation

4. **Experiment with the RAG pipeline:**
   - Modify batch sizes and actor pool sizes
   - Adjust GPU allocations based on your hardware
   - Try different embedding models or LLMs

---

## Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ray Data User Guide](https://docs.ray.io/en/latest/data/data.html)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
