import torch
import chardet
import requests
import numpy as np

from pydantic import BaseModel

import uvicorn
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
par_en_tokenizer, par_en_model = None, None
device = 'cpu'

# Load model and tokenizer
@app.on_event("startup")
def load_model():
    global atd_model, atd_tokenizer, par_model, par_tokenizer, en_tokenizer, en_model, par_en_tokenizer, par_en_model
    atd_tokenizer = BertTokenizer.from_pretrained(ATD_MODEL_NAME)
    atd_model = BertForSequenceClassification.from_pretrained(ATD_MODEL_NAME)
    atd_model.load_state_dict(torch.load(ATD_MODEL_PATH))

    en_tokenizer = AutoTokenizer.from_pretrained("akshayvkt/detect-ai-text")
    en_model = AutoModelForSequenceClassification.from_pretrained("akshayvkt/detect-ai-text")

    par_model = T5ForConditionalGeneration.from_pretrained(PAR_MODEL_NAME)
    par_tokenizer = T5Tokenizer.from_pretrained(PAR_MODEL_NAME)

    par_en_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
    par_en_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')

    atd_model.eval()
    par_model.eval()

    if torch.cuda.is_available():
        atd_model.cuda()
        par_model.cuda()
        par_en_model.cuda()
        device = 'cuda'


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


def paraphrase_single(text, tokenizer, model, lang='en', num_sentences=3, beams=5, grams=4):
    if lang != 'en':
        x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
        max_size = int(x.input_ids.shape[1] * 1.5 + 10)

        generated_sentences = []
        for i in range(num_sentences):
            out = model.generate(**x, encoder_no_repeat_ngram_size=grams, do_sample=True, num_beams=beams,
                                 max_length=max_size, no_repeat_ngram_size=4, )
            generated_sentences.append(tokenizer.decode(out[0], skip_special_tokens=True))

        return generated_sentences


    encoding = tokenizer.encode_plus(text, padding=True, truncation=True, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=int(int(input_ids.shape[1] * 1.5 + 10)),
        top_k=120,
        top_p=0.98,
        beams=beams,
        num_return_sequences=num_sentences
    )

    final_outputs = []
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != text.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    return final_outputs

    # return tokenizer.decode(out[0], skip_special_tokens=True)


def inference_single(text, num_paraphrases=3):
    res = chardet.detect(text.encode('cp1251'))
    lang = 'en'

    cur_model, cur_tokenizer = en_model, en_tokenizer
    if res['language'] == 'Russian' and res['confidence'] > 0.95:
        cur_model, cur_tokenizer = atd_model, atd_tokenizer
        lang = 'ru'

    sec_model, sec_tokenizer = par_en_model, par_en_tokenizer
    if res['language'] == 'Russian' and res['confidence'] > 0.95:
        sec_model, sec_tokenizer = par_model, par_tokenizer

    basic_prob = compute_prob_single(text, cur_tokenizer, cur_model)
    paraphrases = paraphrase_single(text, sec_tokenizer, sec_model, lang=lang, num_sentences=num_paraphrases)
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

    res = chardet.detect(text.encode('cp1251'))
    if res['language'] == 'Russian' and res['confidence'] > 0.95:
        prob = compute_prob_single(text, atd_tokenizer, atd_model)
    else:
        prob = compute_prob_single(text, en_tokenizer, en_model)

    return ProbInferenceResponse(prob=prob)


@app.post('/inference_change', response_model=InferenceResponse)
def inference_change(request: InferenceRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required")

    prob, paraphrases, probabilities = inference_single(text)
    print(text, prob, paraphrases, probabilities )
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