# Knowledge Distillation and Model Optimization Pipeline

This repository demonstrates a complete pipeline for optimizing deep learning models through:
- Teacher Model Fine-tuning
- Knowledge Distillation to a Student Model
- Student Model Fine-tuning
- Post-training Quantization

The goal is to achieve a lightweight model that maintains strong performance, ideal for deployment in resource-constrained environments.

---

## Repository Structure

```plaintext
.
├── Finetuning_Teacher.ipynb         # Fine-tune the teacher model on your dataset
├── Distillation_on_student_model.ipynb  # Train a lightweight student model via distillation
├── Finetuning_Student.ipynb         # Further fine-tune the distilled student model
├── Post_Training_Quantisation_on_Student.ipynb  # Apply quantization for model compression
└── README.md                        # This file

