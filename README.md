## PyTorch for Deep Learning – Personal Implementation

This repository contains my personal implementations and experiments following the “PyTorch for Deep Learning”￼ course by Daniel Bourke. My goal is to replicate all course notebooks from scratch, implement milestone projects, and develop modular, production-ready PyTorch pipelines.

This project demonstrates a solid foundation in PyTorch, deep learning, transfer learning, Vision Transformers (ViT), and model deployment.


Live Demo (Food Vision Big): https://huggingface.co/spaces/prithviraj-maurya/food-vision-big￼


⸻

🌟 Highlights

	•	Reimplemented all course notebooks in a clean, modular GitHub repository

	•	Completed Milestone 08: Paper Replication

	•	Replicated “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale”

	•	Built ViT from scratch and with PyTorch built-in transformer blocks

	•	Achieved ~93–94% test accuracy over 10 epochs

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1     | 0.7663     | 0.7188    | 0.5436    | 0.8769   |
| 2     | 0.3436     | 0.9453    | 0.3257    | 0.8977   |
| 3     | 0.2068     | 0.9492    | 0.2698    | 0.9186   |
| 4     | 0.1556     | 0.9609    | 0.2414    | 0.9186   |
| 5     | 0.1243     | 0.9727    | 0.2271    | 0.8977   |
| 6     | 0.1210     | 0.9766    | 0.2122    | 0.9280   |
| 7     | 0.0933     | 0.9766    | 0.2341    | 0.8883   |
| 8     | 0.0793     | 0.9844    | 0.2268    | 0.9081   |
| 9     | 0.1084     | 0.9883    | 0.2064    | 0.9384   |
| 10    | 0.0646     | 0.9922    | 0.1795    | 0.9176   |

	•	Built modular PyTorch training pipelines:
	•	engine.py: train_step, test_step, train functions
	•	train.py: CLI-based training orchestration
	•	helper_functions.py: Plotting, metrics, utility functions
	•	Implemented Transfer Learning using:
	•	EfficientNet-B2 for Food Vision Big (101 classes)
	•	ViT-B/16 for image classification
	•	Compared accuracy, speed, and model size between EfficientNet-B2 and ViT models
	•	Created and deployed a Gradio app on Hugging Face Spaces:

⸻

📚 Course Progress

Module	Status
00. PyTorch Fundamentals	✅
01. PyTorch Workflow	✅
02. PyTorch Neural Network Classification	✅
03. PyTorch Computer Vision	✅
04. PyTorch Custom Datasets	✅
05. PyTorch Going Modular	✅
06. PyTorch Transfer Learning	✅
07. PyTorch Experiment Tracking	✅
08. PyTorch Paper Replicating	✅
09. PyTorch Model Deployment	✅


⸻

🛠 Key Skills & Concepts

	•	PyTorch Fundamentals: Tensors, computational graphs, autograd

	•	Neural Networks: torch.nn, custom layers, classification, regression

	•	Data Handling: torch.utils.data.Dataset, DataLoader, custom pipelines

	•	Training & Evaluation: Loss functions, optimizers, metrics, GPU acceleration

	•	Transfer Learning: EfficientNet-B2, ViT-B/16, freezing layers, classifier heads

	•	Experiment Tracking: Metrics logging, reproducibility, modular training scripts

	•	Deployment: Gradio apps, Hugging Face Spaces

⸻

🗂 Repository Structure

```
going_modular/
├── engine.py            # train_step, test_step, train functions
├── vit.py               # Vision Transformer from scratch
├── train.py             # CLI-based training orchestration
├── helper_functions.py  # Utilities (plotting, metrics)
notebooks/               # Recreated course notebooks

```
⸻

⚡ Installation

#### Create a virtual environment (recommended)
```
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn gradio
```

⸻

🔗 Links

	•	GitHub Repo: https://github.com/prithviraj-maurya/pytorch_for_deep_learning_ztm_course￼

	•	Live Demo (Food Vision Big): https://huggingface.co/spaces/prithviraj-maurya/food-vision-big￼

	•	Course Resource: learnpytorch.io￼

⸻