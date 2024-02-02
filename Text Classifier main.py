from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

# Set your Hugging Face API token (replace "YOUR_API_TOKEN" with your actual token)
os.environ["HF_HOME"] = os.path.expanduser("~/.huggingface")
os.environ["HF_HOME"] = os.path.expanduser("~/.huggingface")
os.environ["HF_HOME"] = os.path.expanduser("~/.huggingface")
os.environ["HF_HOME"] = os.path.expanduser("~/.huggingface")

model_name = "badalsahani/text-classification-multi"

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class_names = ["Biology",  "Business Studies", "Chemistry", "Computer Science", "Economics",  "English Literature",  "Geography", "History", "Mathematics", "Physics", "Political Science", "Psychology", "Sociology"] 

# Take input
text = input("Enter the text you want to classify: ")

# Use the tokenizer to preprocess the input text
inputs = tokenizer(text, return_tensors="pt")

# Forward pass through the model
outputs = model(**inputs)

# Apply softmax to get predicted probabilities
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get the predicted class index
predicted_class = torch.argmax(probs, dim=-1).item()
predicted_probability = probs[0, predicted_class].item()

# Print the predicted class and probabilities
print(f"\nPredicted Class: {class_names[predicted_class]} ({predicted_probability*100:.2f} %) \n")
print("Predicted Probabilities: \n")
for class_name, prob in zip(class_names,probs[0].tolist()):
    print(f"{class_name}: {prob*100:.2f} %")
