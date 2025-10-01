import gradio as gr
from textSummarizer.pipeline.prediction import PredictionPipeline

pipeline = PredictionPipeline()

def summmarize_text(text: str) ->str:
    return pipeline.predict(text)


demo = gr.Interface(
    fn = summmarize_text,
    inputs = gr.Textbox(lines=10,placeholder="Paste text here..."),
    outputs=gr.Textbox(label="Summary"),
    title="Text Summarizer (Pegasus Samsum)",
    description="This app summarizes dialogues using fine-tuned Pegasus model."
)


if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)