from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("badalsahani/text-classification-multi", token=True)

tokenizer = AutoTokenizer.from_pretrained("badalsahani/text-classification-multi", token=True)

inputs = tokenizer("I love AutoTrain", return_tensors="pt")

outputs = model(**inputs)
