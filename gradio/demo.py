import gradio as gr

gr.Interface.load("huggingface/allenai/longformer-large-4096-finetuned-triviaqa", examples=[
  ["My name is Wolfgang and I live in Berlin", "Where do I live?"],
  ["The capital of France is Paris", "What is the capital of France?"]
]).launch()
