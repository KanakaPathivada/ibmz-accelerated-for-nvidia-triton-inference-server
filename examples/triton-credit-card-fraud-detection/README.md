# Credit Card Fraud Detection with IBM Z Accelerated for NVIDIA Triton™ Inference Server

This exmaple provides **end-to-end examples** for deploying and benchmarking machine learning and deep learning models for **credit card fraud detection** using the [IBM Z Accelerated for NVIDIA Triton Inference Server](https://github.com/IBM/ibmz-accelerated-for-nvidia-triton-inference-server). It demonstrates how to serve models using three major backends:

- IBM Snap ML Backend (Snap ML Boosting Machine/ LightGBM / XGBoost)
- ONNX-MLIR Backend  
- PyTorch Backend  

Each notebook demonstrates how to prepare a model, deploy it with Triton, and profile it using Triton **Model Analyzer** for production-level performance evaluation.

---

## Folder Structure

```
.
├── CCF_LSTM_Trained_Model_Inputs                # Pre-trained model and sample input
│   └──input_data.json
├── ibmsnapml-backend                            # Snap ML backend examples (LightGBM, XGBoost)
│   ├── CreditCardFraud_Detection_LightGBM_Example.ipynb
│   ├── CreditCardFraud_Detection_SnapML_BoostingMachine_Example.ipynb
│   └── CreditCardFraud_Detection_XGBoost_Example.ipynb
├── onnx-mlir-backend                            # ONNX-MLIR backend example for LSTM
│   └── Credit_Card_Fraud_Detection_LSTM_TritonIS_ONNXMLIR_backend.ipynb
├── pytorch-backend                              # PyTorch backend example for LSTM
│   └── Credit_Card_Fraud_Detection_LSTM_TritonIS_PyTorch_backend.ipynb
└── README.md
```

---

## Project Overview
The goal is to detect fraudulent transactions using an LSTM or tree-based classification model and deploy it in a production-grade inference environment with Triton IS. Each example:

1. Covers pre-processing, model conversion (if needed), and deployment
2. Implements a live inference API with Triton IS
3. Benchmarks performance using Triton Model Analyzer
4. Optionally explores acceleration on IBM Z NNPA, CPU, or Snap ML

---

## Supported Backends & Notebooks

### PyTorch Backend
**Notebook:** 
```
pytorch-backend/Credit_Card_Fraud_Detection_LSTM_TritonIS_PyTorch_backend.ipynb
```
Demonstrates serving a PyTorch-trained LSTM model via Triton with detailed steps including:
- Serves an LSTM model using PyTorch backend in Triton
- Step-by-step setup of model repository & config
- Triton server deployment and inference


### Prerequisites to Deploy the Model with ONNX-MLIR and PyTorch Backend
#### Model Summary
This model is designed to detect fraudulent credit card transactions using a sequential LSTM architecture.
- Training: Performed using PyTorch, following the workflow provided in this [GitHub repository](https://github.com/IBM/ibmz-accelerated-for-pytorch/tree/main/samples/credit-card-fraud)
- Preprocessing: Includes encoding of categorical fields, normalization of time-based features, and scaling of transaction amounts
- Model Architecture: LSTM with sequence length 7 and hidden size 200
- Export: Trained model is saved in .pt format and exported to ONNX for deployment with ONNX-MLIR or PyTorch backends

**NOTE:** To train the model and save it in the required `.pt` format, refer to the GitHub repo linked above. This is the recommended starting point for deploying with ONNX-MLIR or PyTorch.

### ONNX-MLIR Backend
**Notebook:** 
```
onnx-mlir-backend/Credit_Card_Fraud_Detection_LSTM_TritonIS_ONNXMLIR_backend.ipynb
```
This example demonstrates how to:
- Convert a PyTorch-trained LSTM model into ONNX format
- Compile it using the [IBM Z Deep Learning Compiler](https://github.com/IBM/zDLC)
- Deploy the compiled model using the ONNX-MLIR backend with Triton Inference Server
- Run inference on both CPU and NNPA (Neural Network Processing Assist) available on IBM Z
- Profile performance using Triton Model Analyzer

#### Key Highlights:
1. End-to-end deployment with onnx-mlir backend
2. Benchmarks include latency, throughput, and resource utilization
3. Outputs HTML and CSV reports from Triton Model Analyzer


### IBM Snap ML Backend
Notebooks for LightGBM, SnapBoost, and XGBoost:
```
CreditCardFraud_Detection_LightGBM_Example.ipynb

CreditCardFraud_Detection_SnapML_BoostingMachine_Example.ipynb

CreditCardFraud_Detection_XGBoost_Example.ipynb
```

#### Highlights:
- Uses Snap ML-optimized tree models for accelerated inference
- Integrated with Triton SnapML backend for accelerated inference
- Optimized for structured fraud detection workloads on IBM Z


---

## Performance Analysis with Model Analyzer

### Triton Model Analyzer helps profile:
- Inference latency
- Throughput
- CPU/memory utilization

### Supported outputs:
- HTML summary reports
- CSV raw metrics

Refer to the [Triton Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/model_analyzer.html) documentation for all config options.

**Note:** Triton containers till v1.4.0 may not include pyyaml which is required for YAML-based config. You may need:
```
pip install pyyaml
```

---

## Setup Instructions
1. Clone the Repository
```
git clone https://github.com/IBM/ai-on-z-triton-is-examples.git
cd ai-on-z-triton-is-examples/triton-credit-card-fraud-detection
```
2. Launch Jupyter Notebook
```
jupyter notebook
```
Browse through the notebooks under:
1. `ibmsnapml-backend/`
2. `onnx-mlir-backend/`
3. `pytorch-backend/`

---

## Reference Links
1. https://github.com/IBM/ibmz-accelerated-for-nvidia-triton-inference-server
2. https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-nvidia-triton-inference-server.html
3. https://github.com/IBM/zDLC
4. https://ibm.github.io/ibm-z-oss-hub/containers/zdlc.html
5. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/model_analyzer.html
6. https://github.com/IBM/ibmz-accelerated-for-pytorch/tree/main/samples/credit-card-fraud