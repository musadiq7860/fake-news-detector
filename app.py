import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "Musadiq7860/fake-news-detector"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = outputs.logits.softmax(dim=-1)
        pred    = probs.argmax().item()
        confidence = probs.max().item() * 100
    label = "🔴 FAKE NEWS" if pred == 1 else "🟢 REAL NEWS"
    return f"{label}\nConfidence: {confidence:.2f}%"

gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Paste a news headline or article here..."),
    outputs=gr.Textbox(label="Prediction"),
    title="Fake News Detector",
    description="Fine-tuned DistilBERT model on 10,000 news articles."
).launch()
