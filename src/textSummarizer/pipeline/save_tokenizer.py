from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")

tokenizer.save_pretrained("artifacts/model/content/pegasus-samsum-model")

