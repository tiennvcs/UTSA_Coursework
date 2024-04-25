from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
import numpy as np


test_dataset = load_dataset("imagenet-1k", cache_dir="~/.cache/huggingface/datasets", split='train')


random_idx = np.random.randint(100)
random_test_image = test_dataset[random_idx]["image"]

print("Hello")

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(random_test_image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
print("Idx: {}".format(random_idx))
predicted_label = model.config.id2label[logits.argmax(-1).item()]
true_label = test_dataset["label"][random_idx]
print("Predicted label: {}".format(predicted_label))
print("True label: {}".format(true_label))