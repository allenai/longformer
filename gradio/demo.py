import gradio as gr

description = "demo for LongformerQA. To use it, simply add your text or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2004.05150'>Longformer: The Long-Document Transformer</a> | <a href='https://github.com/allenai/longformer'>Github Repo</a></p>"

gr.Interface.load("huggingface/allenai/longformer-large-4096-finetuned-triviaqa", examples=[
  ["My name is Wolfgang and I live in Berlin", "Where do I live?"],
  ["The capital of France is Paris", "What is the capital of France?"]
]).launch()
