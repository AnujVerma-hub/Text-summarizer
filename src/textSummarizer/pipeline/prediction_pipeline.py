from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import os

class Predictionpipeline:
    def __init__(self, model_dir ="artifacts/content/pegasus-samsum-model"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, from_safetensors=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.summarizer = pipeline("summarization", model=self.model,tokenizer=self.tokenizer)
        


    def summarize(self, text, max_len=128, min_len=20):

        """Summarize the given text using pegasus model
        Args:
          text(str): The input text to summarize
          max_len(int): Maximum length of summary
          imn_len(int): Minimum length of summary
        """
        result = self.summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    
        return result[0]['summary_text']
    