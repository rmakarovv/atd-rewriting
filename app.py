import torch
import chardet
import requests
import numpy as np

from pydantic import BaseModel

import uvicorn
from fastapi import Request
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from transformers import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bs4 import BeautifulSoup

# Disable warnings
logging.set_verbosity_warning()

# Constants
ATD_MODEL_PATH = "atd_rubert.pt"
ATD_MODEL_NAME = 'cointegrated/rubert-tiny2'
PAR_MODEL_NAME = 'cointegrated/rut5-base-paraphraser'

description = open('description.md').read()

# Initialize FastAPI app
app = FastAPI(
    title="AI детектор",
    description=description,
    docs_url='/',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

atd_model, atd_tokenizer, par_model, par_tokenizer, en_tokenizer, en_model = None, None, None, None, None, None


# Load model and tokenizer
@app.on_event("startup")
def load_model():
    global atd_model, atd_tokenizer, par_model, par_tokenizer, en_tokenizer, en_model
    atd_tokenizer = BertTokenizer.from_pretrained(ATD_MODEL_NAME)
    atd_model = BertForSequenceClassification.from_pretrained(ATD_MODEL_NAME)
    atd_model.load_state_dict(torch.load(ATD_MODEL_PATH))

    en_tokenizer = AutoTokenizer.from_pretrained("akshayvkt/detect-ai-text")
    en_model = AutoModelForSequenceClassification.from_pretrained("akshayvkt/detect-ai-text")

    par_model = T5ForConditionalGeneration.from_pretrained(PAR_MODEL_NAME)
    par_tokenizer = T5Tokenizer.from_pretrained(PAR_MODEL_NAME)

    atd_model.eval()
    par_model.eval()

    if torch.cuda.is_available():
        atd_model.cuda()
        par_model.cuda()


class InferenceRequest(BaseModel):
    text: str


class ProbInferenceResponse(BaseModel):
    prob: float


class InferenceResponse(BaseModel):
    prob: float
    paraphrases: list
    probabilities: list


class UrlInferenceResponse(BaseModel):
    text: str
    prob: float
    paraphrases: list
    probabilities: list


def softmax(x):
    """Compute softmax over a list of values."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def compute_prob_single(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt')
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    with torch.no_grad():
        pred = model(**tokens)

    prob = pred.logits[0].cpu().numpy()
    prob = softmax(prob)
    return int(prob[1] * 100)


def paraphrase_single(text, tokenizer, model, beams=5, grams=4, do_sample=True):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams, do_sample=do_sample, num_beams=beams,
                         max_length=max_size, no_repeat_ngram_size=4, )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def inference_single(text, num_paraphrases=3):
    cur_model, cur_tokenizer = atd_model, atd_tokenizer
    if chardet.detect(text.encode('cp1251'))['language'] != 'Russian':
        cur_model, cur_tokenizer = en_model, en_tokenizer

    basic_prob = compute_prob_single(text, cur_tokenizer, cur_model)
    paraphrases = [paraphrase_single(text, par_tokenizer, par_model) for i in range(num_paraphrases)]
    probabilities = [compute_prob_single(par, cur_tokenizer, cur_model) for par in paraphrases]

    both = list(zip(paraphrases, probabilities))
    both.sort(key=lambda x: x[1])
    paraphrases, probabilities = zip(*both)

    return basic_prob, list(paraphrases), list(probabilities)


@app.post("/inference", response_model=InferenceResponse)
def get_inference(request: InferenceRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required")

    prob, paraphrases, probabilities = inference_single(text)
    return InferenceResponse(prob=prob, paraphrases=paraphrases, probabilities=probabilities)


@app.post('/inference_prob', response_model=ProbInferenceResponse)
def inference_probability(request: InferenceRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required")

    if chardet.detect(text.encode('cp1251'))['language'] != 'Russian':
        prob = compute_prob_single(text, en_tokenizer, en_model)
    else:
        prob = compute_prob_single(text, atd_tokenizer, atd_model)

    return ProbInferenceResponse(prob=prob)


@app.post('/inference_change', response_model=InferenceResponse)
def inference_change(request: InferenceRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required")

    prob, paraphrases, probabilities = inference_single(text)
    return InferenceResponse(prob=prob, paraphrases=paraphrases, probabilities=probabilities)


@app.post("/analyze_url", response_model=UrlInferenceResponse)
async def analyze_url(request: InferenceRequest):
    url = request.text.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)

        print(text)

        if not text:
            raise HTTPException(status_code=400, detail="No text found on the page")

        # TODO: update the context length
        if len(text) > 500:
            text = text[:500]

        prob, paraphrases, probabilities = inference_single(text, num_paraphrases=1)
        return UrlInferenceResponse(text=text, prob=prob, paraphrases=paraphrases, probabilities=probabilities)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)