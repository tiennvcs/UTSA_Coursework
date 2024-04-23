from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset


from datasets import list_datasets
datasets_list = list_datasets()
print(datasets_list)

dataset = load_dataset("imagenet-1k")

test_image = dataset["test"]["image"][10]

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(test_image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
