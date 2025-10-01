from textSummarizer.pipeline.prediction_pipeline import Predictionpipeline

if __name__=="__main__":

    predictor = Predictionpipeline()


    text = "Yesterday, I had a very hectic day at work. I woke up late because my alarm clock didn't ring, amd I had only 20 minutes to get ready for an important meeting with my manager. While rushing, I spilled coffee on my shirt and had to quickly change, which made me even later."
    
    summary = predictor.summarize(text)
    print("\n===SUMMARY===")
    print(summary)