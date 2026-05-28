"""
Api.py — FastAPI-сервер для анализа тональности русскоязычных комментариев.

Запуск:
    python Api.py
    или
    uvicorn Api:app --reload --host 0.0.0.0 --port 8000

Эндпоинты:
    GET  /                       — список всех эндпоинтов (справка)
    GET  /models                 — доступные модели
    GET  /stats/overview         — статистика датасета
    POST /predict                — классификация одного текста
    POST /predict/batch          — классификация нескольких текстов
    GET  /health                 — проверка работоспособности
    GET  /docs                   — Swagger UI (автогенерация FastAPI)
"""

import re
import json
import joblib
import numpy as np
import pandas as pd
import uvicorn
import pymorphy3
import nltk

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from nltk.corpus import stopwords
from pathlib import Path
from typing import Optional

# ─── Пути ────────────────────────────────────────────────────────────────────
APP_DIR   = Path(__file__).resolve().parent
ROOT_DIR  = APP_DIR.parent
MODELS_DIR = APP_DIR / "Models"       # папка рядом с Api.py
DATA_DIR   = APP_DIR / "Resource"     # папка с CSV-данными
STATS_FILE = DATA_DIR / "stats_cache.json"

# ─── Глобальные объекты ───────────────────────────────────────────────────────
models: dict      = {}
tfidf             = None
lemma_cache: dict = {}
stats_cache: dict = {}

morph = pymorphy3.MorphAnalyzer()
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("russian"))
STOP_WORDS.update({
    "это", "так", "вот", "ну", "да", "нет", "ещё", "уже", "просто",
    "очень", "тоже", "только", "всё", "всегда", "когда", "сейчас",
    "http", "https", "com", "org", "rt", "via", "amp",
})

# ─── Предобработка текста ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Полный пайплайн предобработки: lowercase → очистка → лемматизация."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|@\w+|#\w+", " ", text)   # URL, @mentions, #теги
    words = re.findall(r"[а-яёa-z]+", text)
    result = []
    for w in words:
        if len(w) < 3 or w in STOP_WORDS:
            continue
        if w not in lemma_cache:
            lemma_cache[w] = morph.parse(w)[0].normal_form
        lemma = lemma_cache[w]
        if lemma not in STOP_WORDS and len(lemma) >= 3:
            result.append(lemma)
    return " ".join(result)

# ─── Загрузка моделей ─────────────────────────────────────────────────────────
def load_models():
    global tfidf, models, lemma_cache

    # TF-IDF векторизатор
    for fname in ["tfidf_vectorizer.pkl", "tfidf_vectorizer.prek"]:
        p = MODELS_DIR / fname
        if p.exists():
            tfidf = joblib.load(p)
            print(f"✔ Векторизатор загружен: {fname}")
            break

    # Лемматизационный кэш (ускоряет предобработку)
    cache_path = MODELS_DIR / "lemma_cache.pkl"
    if cache_path.exists():
        lemma_cache = joblib.load(cache_path)
        print(f"✔ Кэш лемм: {len(lemma_cache):,} записей")

    # Модели классификации
    model_files = {
        "Logistic Regression":       "model_logistic_regression.pkl",
        "Naive Bayes":               "model_naive_bayes.pkl",
        "Linear SVC":                "model_linear_svc.pkl",
        "Финальная (self-trained)":  "final_model.pkl",
    }
    for name, fname in model_files.items():
        p = MODELS_DIR / fname
        if p.exists():
            models[name] = joblib.load(p)
            print(f"✔ Модель загружена: {name}")

    if not models:
        print("⚠  Модели не найдены. Проверьте папку Models/")
    if tfidf is None:
        print("⚠  Векторизатор не найден. Проверьте папку Models/")

# ─── Статистика датасета ──────────────────────────────────────────────────────
def compute_stats() -> dict:
    """Считает статистику по CSV один раз, кэширует в JSON."""
    if STATS_FILE.exists():
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("✔ Статистика загружена из кэша")
        return data

    # Ищем neg.csv / pos.csv
    neg_path = DATA_DIR / "neg.csv"
    pos_path = DATA_DIR / "pos.csv"
    if not (neg_path.exists() and pos_path.exists()):
        return {"error": "Файлы neg.csv / pos.csv не найдены"}

    print("Вычисляю статистику датасета...")
    COLS = ["tweet_id", "timestamp", "author", "text", "sentiment",
            "replies", "retweets", "likes",
            "user_followers", "user_friends", "user_statuses", "listed_count"]

    df_neg = pd.read_csv(neg_path, sep=";", header=None, names=COLS,
                         on_bad_lines="skip", quotechar='"')
    df_pos = pd.read_csv(pos_path, sep=";", header=None, names=COLS,
                         on_bad_lines="skip", quotechar='"')
    df = pd.concat([df_neg, df_pos], ignore_index=True)
    df["label"] = (df["sentiment"] == 1).astype(int)
    df["text"]  = df["text"].fillna("")
    df["text_len_chars"] = df["text"].str.len()
    df["text_len_words"] = df["text"].str.split().str.len()

    total       = int(len(df))
    neg_count   = int((df.label == 0).sum())
    pos_count   = int((df.label == 1).sum())
    avg_chars   = float(df["text_len_chars"].mean().round(1))
    avg_words   = float(df["text_len_words"].mean().round(1))
    median_words = float(df["text_len_words"].median())

    # Гистограмма длин (сэмпл 3000 точек для фронта)
    sample_lens = df["text_len_words"].dropna().sample(
        n=min(3000, len(df)), random_state=42
    ).astype(int).tolist()

    # Распределение по длинам (бакеты 0-5, 6-10, ...)
    bins = list(range(0, 55, 5))
    hist, edges = np.histogram(df["text_len_words"].dropna(), bins=bins)
    len_distribution = {
        f"{edges[i]}-{edges[i+1]}": int(hist[i])
        for i in range(len(hist))
    }

    result = {
        "total_comments":          total,
        "negative_count":          neg_count,
        "positive_count":          pos_count,
        "negative_pct":            round(neg_count / total * 100, 2),
        "positive_pct":            round(pos_count / total * 100, 2),
        "avg_length_chars":        avg_chars,
        "avg_length_words":        avg_words,
        "median_length_words":     median_words,
        "length_distribution":     len_distribution,
        "word_count_sample":       sample_lens,
        "unique_authors":          int(df["author"].nunique()),
        "avg_likes":               round(float(df["likes"].mean()), 2),
        "avg_retweets":            round(float(df["retweets"].mean()), 2),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print("✔ Статистика вычислена и закэширована")
    return result

# ─── Жизненный цикл приложения ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("Запуск сервера анализа тональности")
    print("=" * 50)
    load_models()
    global stats_cache
    stats_cache = compute_stats()
    print("Сервер готов к работе ✔")
    yield

# ─── Приложение FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "API для анализа тональности русскоязычных комментариев. "
        "Поддерживает несколько классификаторов (Logistic Regression, "
        "Naive Bayes, Linear SVC, финальная self-trained модель)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic-схемы ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    model_name: str = Field(
        default="Финальная (self-trained)",
        description="Название модели. Получить список: GET /models",
        example="Logistic Regression",
    )
    text: str = Field(
        description="Текст комментария для анализа тональности",
        example="Отличный день, всё получилось! Очень доволен.",
        min_length=1,
        max_length=5000,
    )

class BatchPredictRequest(BaseModel):
    model_name: str = Field(
        default="Финальная (self-trained)",
        description="Название модели",
    )
    texts: list[str] = Field(
        description="Список текстов (до 100 штук)",
        min_length=1,
        max_length=100,
        example=["Отличный день!", "Всё сломалось, ужасно."],
    )

# ─── Эндпоинты ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Справка"], summary="Список всех эндпоинтов")
def root():
    """
    Возвращает справку по всем доступным командам API с описанием параметров.
    Также доступна интерактивная документация: GET /docs
    """
    return {
        "service": "Sentiment Analysis API v1.0",
        "description": "Анализ тональности русскоязычных комментариев",
        "endpoints": [
            {
                "method": "GET",
                "path": "/",
                "description": "Эта справка — список всех команд и параметров",
            },
            {
                "method": "GET",
                "path": "/health",
                "description": "Проверка работоспособности сервера и загруженности моделей",
            },
            {
                "method": "GET",
                "path": "/models",
                "description": "Список доступных моделей классификации",
            },
            {
                "method": "GET",
                "path": "/stats/overview",
                "description": "Статистика датасета: кол-во, доли классов, длины текстов и т.д.",
            },
            {
                "method": "POST",
                "path": "/predict",
                "description": "Анализ тональности одного текста",
                "body": {
                    "model_name": "str — название модели (из /models)",
                    "text":       "str — текст комментария (до 5000 символов)",
                },
                "returns": {
                    "sentiment":    "«Позитив» / «Негатив»",
                    "label":        "1 (позитив) / 0 (негатив)",
                    "confidence":   "float 0–1 — уверенность модели",
                    "prob_neg":     "float — вероятность негатива",
                    "prob_pos":     "float — вероятность позитива",
                    "cleaned_text": "str — текст после предобработки",
                    "model_used":   "str — использованная модель",
                },
            },
            {
                "method": "POST",
                "path": "/predict/batch",
                "description": "Анализ тональности массива текстов (до 100 штук)",
                "body": {
                    "model_name": "str — название модели",
                    "texts":      "list[str] — список текстов",
                },
                "returns": "list результатов, аналогичных /predict",
            },
            {
                "method": "GET",
                "path": "/docs",
                "description": "Интерактивная Swagger-документация (Swagger UI)",
            },
        ],
    }


@app.get("/health", tags=["Справка"], summary="Статус сервера")
def health():
    """Возвращает статус загрузки моделей и векторизатора."""
    return {
        "status":          "ok",
        "models_loaded":   list(models.keys()),
        "models_count":    len(models),
        "vectorizer":      "loaded" if tfidf is not None else "NOT LOADED",
        "lemma_cache_size": len(lemma_cache),
        "stats_loaded":    bool(stats_cache and "error" not in stats_cache),
    }


@app.get("/models", tags=["Модели"], summary="Список моделей")
def get_models():
    """Возвращает список доступных моделей классификации."""
    return [{"name": k} for k in models.keys()]


@app.get("/stats/overview", tags=["Статистика"], summary="Статистика датасета")
def overview():
    """
    Возвращает агрегированную статистику обучающего датасета:
    - кол-во комментариев, доля позитива/негатива
    - средняя и медианная длина текстов
    - распределение длин по бакетам
    - кол-во уникальных авторов, среднее число лайков/ретвитов
    """
    if not stats_cache:
        raise HTTPException(status_code=503, detail="Статистика не загружена")
    return stats_cache


@app.post("/predict", tags=["Классификация"], summary="Анализ тональности текста")
def predict(req: PredictRequest):
    """
    Определяет тональность одного комментария.

    - **model_name** — модель из списка `/models`. Если пусто, используется финальная.
    - **text** — текст на русском языке (до 5000 символов).

    Возвращает: класс (Позитив / Негатив), вероятности, уверенность модели.
    """
    _check_ready()

    name = req.model_name or _default_model()
    if name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{name}' не найдена. Доступные: {list(models.keys())}",
        )

    cleaned = clean_text(req.text)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Текст пустой после предобработки")

    return _classify(cleaned, req.text, name)


@app.post("/predict/batch", tags=["Классификация"], summary="Пакетный анализ текстов")
def predict_batch(req: BatchPredictRequest):
    """
    Определяет тональность списка комментариев за один запрос (до 100 текстов).
    Возвращает список результатов в том же порядке, что входные тексты.
    """
    _check_ready()

    name = req.model_name or _default_model()
    if name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{name}' не найдена. Доступные: {list(models.keys())}",
        )
    if len(req.texts) > 100:
        raise HTTPException(status_code=400, detail="Максимум 100 текстов за раз")

    results = []
    for text in req.texts:
        cleaned = clean_text(text)
        if cleaned.strip():
            results.append(_classify(cleaned, text, name))
        else:
            results.append({
                "original_text": text,
                "error": "Текст пустой после предобработки",
            })
    return results


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _check_ready():
    if not models:
        raise HTTPException(status_code=503, detail="Модели не загружены")
    if tfidf is None:
        raise HTTPException(status_code=503, detail="Векторизатор не загружен")


def _default_model() -> str:
    preferred = "Финальная (self-trained)"
    return preferred if preferred in models else next(iter(models))


def _classify(cleaned: str, original: str, model_name: str) -> dict:
    """Внутренняя функция классификации — не дублировать логику."""
    vec   = tfidf.transform([cleaned])
    model = models[model_name]
    label = int(model.predict(vec)[0])

    prob_neg, prob_pos, confidence = None, None, None

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(vec)[0]
        prob_neg   = round(float(proba[0]), 4)
        prob_pos   = round(float(proba[1]), 4)
        confidence = round(float(proba.max()), 4)
    elif hasattr(model, "decision_function"):
        scores     = model.decision_function(vec)[0]
        # Для бинарного SVC decision_function возвращает скаляр
        if np.ndim(scores) == 0:
            prob_pos   = round(float(1 / (1 + np.exp(-scores))), 4)
            prob_neg   = round(1 - prob_pos, 4)
            confidence = round(max(prob_pos, prob_neg), 4)

    return {
        "original_text": original[:200],
        "cleaned_text":  cleaned[:200],
        "sentiment":     "Позитив" if label == 1 else "Негатив",
        "label":         label,
        "prob_neg":      prob_neg,
        "prob_pos":      prob_pos,
        "confidence":    confidence,
        "model_used":    model_name,
    }


# ─── Точка входа ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
