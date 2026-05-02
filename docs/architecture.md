# Procr - Documentation Perception Engine (v2)
## Architecture Design: MinerU 2.5 Pro Integration

### 1. Overview
Procr is the second-generation document perception service for the Antigravity platform. It replaces the multi-stage Surya pipeline with a unified **Vision-Language Model (VLM)** approach using `MinerU-2.5-Pro-2604-1.2B`.

### 2. Core Components

#### A. Unified VLM Model Manager (`procr/app/core/model_manager.py`)
- **Engine**: `vllm-engine` (preferred) or `transformers`.
- **Model**: `opendatalab/MinerU2.5-Pro-2604-1.2B`.
- **Strategy**: Eager loading into VRAM at startup. Estimated VRAM footprint: ~6-8GB on CUDA.
- **Precision**: FP16/BF16.

#### B. The MinerU Adapter (`procr/app/services/adapter.py`)
This is the most critical component. It bridges the gap between MinerU's proprietary JSON and our `DocumentGraph` format.
- **Input**: MinerU `middle.json` (intermediate results).
- **Processing**:
    - **Layout Mapping**: Map `para_blocks` to logical sections.
    - **Atomic Citations**: Extract `spans` from the JSON. Each span contains a `bbox` and its recognized text.
    - **Math Extraction**: Capture inline and block formulas directly as LaTeX.
    - **Table Extraction**: Capture tables as clean Markdown/HTML structures.
- **Output**: A standardized `DocumentGraph` JSON that the Node.js server can consume.

#### C. High-Fidelity Prompting (`procr/app/services/prompter.py`)
- Logic to inject `[[nodeId]]` markers into the final extracted text.
- Ensures that even complex structures like tables have internal markers for precise grounding.

### 3. Pipeline Flow
1. **Request**: Node.js sends a Base64 image of a PDF page.
2. **Execution**: 
   - VLM processes the image.
   - MinerU-VL-Utils extracts structural data.
3. **Adaptation**: The `MinerU Adapter` transforms the structural JSON into `DocumentGraph` nodes.
4. **Response**: Returns a structured JSON containing nodes (text, images, tables, math) with normalized bounding boxes.

### 4. Technical Specifications
- **Framework**: FastAPI (v0.115.0+).
- **Inference Library**: `mineru-vl-utils` (v0.2.6).
- **Python Version**: 3.10 - 3.12.
- **Pinned Dependencies**: `magic-pdf[full]==1.1.0`, `torch==2.4.0`, `transformers==4.45.0`.

### 5. Transition Strategy
- **Co-existence**: Procr will live alongside PyOCR (v1). 
- **Rollout**: The Node.js server can toggle between `PythonOCRService` (v1) and `ProcrService` (v2) based on configuration or file complexity.
