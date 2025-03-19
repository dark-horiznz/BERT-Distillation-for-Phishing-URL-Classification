# Knowledge Distillation and Model Optimization Pipeline

This repository demonstrates a complete pipeline for optimizing deep learning models through a series of techniques designed to create efficient, deployment-ready models without sacrificing performance.

## Overview

The pipeline consists of four main stages:
1. **Teacher Model Fine-tuning** - Optimize a large pre-trained model for the target task
2. **Knowledge Distillation** - Transfer knowledge from teacher to a smaller student model
3. **Student Model Fine-tuning** - Compare Student DIstillation with Standard Fine-Tuning
4. **Post-training Quantization** - Compress the model for deployment in resource-constrained environments

## Repository Structure

```
.
├── Finetuning_Teacher.ipynb         # Fine-tune the teacher model on your dataset
├── Distillation_on_student_model.ipynb  # Train a lightweight student model via distillation
├── Finetuning_Student.ipynb         # Further fine-tune the distilled student model
├── Post_Training_Quantisation_on_Student.ipynb  # Apply quantization for model compression
└── README.md                        # This file
```

## Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score | Notes |
|-------|----------|-----------|--------|----------|-------|
| Teacher (BERT base) | 0.8711 | 0.9073 | 0.8267 | 0.8651 | Full-sized model |
| Student (Distilled) | 0.9267 | 0.9486 | 0.9022 | 0.9248 | Smaller architecture |
| Student (Fine-tuned) | 0.8620 | - | - | - | AUC: 0.946 |
| Student (Quantized) | 0.9156 | 0.9401 | 0.8908 | 0.9148 | 4-bit quantization |

## Pipeline Details

### 1. Teacher Model Fine-tuning

We start by fine-tuning a BERT base model on our classification task:

```python
model_path = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}
model = AutoModelForSequenceClassification.from_pretrained(model_path,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,)
```

### 2. Knowledge Distillation

Knowledge distillation transfers the knowledge from the teacher model to a smaller student model:

```python
from transformers import DistilBertForSequenceClassification, DistilBertConfig
config = DistilBertConfig(n_heads=8, n_layers=4)
student_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
    config=config,).to(device)
```

The distillation process uses a combination of soft targets (teacher logits) and hard targets (true labels):

```python
def distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha):
    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
    student_soft = nn.functional.log_softmax(student_logits / temperature, dim=1)
    distill_loss = nn.functional.kl_div(student_soft, soft_targets, reduction='batchmean') * (temperature ** 2)
    hard_loss = nn.CrossEntropyLoss()(student_logits, true_labels)
    loss = alpha * distill_loss + (1.0 - alpha) * hard_loss
    return loss
```

### 3. Student Model Fine-tuning

After distillation, we further fine-tune the student model on the target dataset to enhance performance.

### 4. Post-training Quantization

Finally, we apply 4-bit quantization using the BitsAndBytes library:

```python
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model_nf4 = AutoModelForSequenceClassification.from_pretrained(model_id, 
                                                             device_map=device, 
                                                             quantization_config=nf4_config)
```

## Usage

1. Start by running `Finetuning_Teacher.ipynb` to create a well-tuned teacher model
2. Run `Distillation_on_student_model.ipynb` to transfer knowledge to the student model
3. Run `Finetuning_Student.ipynb` for additional fine-tuning of the student model
4. Run `Post_Training_Quantisation_on_Student.ipynb` to create a deployment-ready quantized model

## Requirements

- PyTorch
- Transformers
- BitsAndBytes
- Scikit-learn
- NumPy

## Conclusion

This pipeline demonstrates how to effectively compress models through knowledge distillation and quantization while maintaining or even improving performance. The final quantized student model is significantly smaller than the original teacher model while maintaining comparable accuracy metrics, making it suitable for deployment in resource-constrained environments.
