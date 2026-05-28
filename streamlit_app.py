"""
streamlit_app.py — Графический интерфейс для анализа тональности комментариев.

Запуск:
    streamlit run streamlit_app.py

Требует запущенного Api.py на порту 8000.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Конфигурация страницы ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Анализ тональности",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ─── Стили ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 16px 20px;
    border-left: 4px solid #4e89e8;
}
.pos-badge {
    background: #d4edda; color: #155724;
    padding: 4px 14px; border-radius: 20px;
    font-weight: bold; font-size: 1.1em;
}
.neg-badge {
    background: #f8d7da; color: #721c24;
    padding: 4px 14px; border-radius: 20px;
    font-weight: bold; font-size: 1.1em;
}
.conf-bar-wrap { margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Хелперы для запросов ─────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def api_get(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{API_URL}/{endpoint}", params=params, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}/{endpoint}", json=payload, timeout=20)
        return r.json(), r.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


# ─── Загружаем данные при старте ─────────────────────────────────────────────
stats       = api_get("stats/overview")
models_list = api_get("models") or []
health      = api_get("health") or {}

model_names = [m["name"] for m in models_list]

# ─── Боковая панель ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💬 Тональность")
    st.caption("Анализ русскоязычных твитов")
    st.divider()

    # Статус API
    if stats:
        st.success("API онлайн ✔")
    else:
        st.error("API недоступен")
        st.info("Запустите: `python Api.py`")

    st.divider()
    st.subheader("Навигация")
    page = st.radio(
        "Раздел:",
        ["🔍 Классификатор", "📊 Статистика датасета", "📘 Справка по API"],
        label_visibility="collapsed",
    )

    if model_names:
        st.divider()
        st.subheader("Модель по умолчанию")
        default_model = st.selectbox(
            "Модель:", model_names, label_visibility="collapsed"
        )
    else:
        default_model = None

    st.divider()
    st.caption("Датасет: русскоязычные твиты\nКлассы: позитив / негатив")

# ─── Заголовок ───────────────────────────────────────────────────────────────
st.title("💬 Анализ тональности комментариев")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 1: КЛАССИФИКАТОР
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Классификатор":
    st.subheader("Определение тональности текста")

    tab_single, tab_batch = st.tabs(["Один текст", "Пакетный анализ"])

    # ── Один текст ────────────────────────────────────────────────────────────
    with tab_single:
        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            st.markdown("**Введите комментарий:**")
            input_text = st.text_area(
                "text",
                placeholder="Например: Отличный день, всё получилось! Очень доволен.",
                height=160,
                label_visibility="collapsed",
            )

            # Примеры для быстрой вставки
            with st.expander("📌 Примеры текстов"):
                examples = {
                    "😊 Позитивный":  "Наконец-то сдал экзамен, теперь могу выдохнуть! Очень рад результату.",
                    "😔 Негативный":  "Снова задержали поезд, уже второй раз за неделю. Ужасный сервис.",
                    "😤 Резкий":      "Ужасный фильм, потратил два часа впустую. Никому не советую.",
                    "😄 Радостный":   "Отлично провели время с семьёй! Лучший выходной за долгое время.",
                }
                for label, ex in examples.items():
                    if st.button(label, use_container_width=True):
                        st.session_state["example_text"] = ex

            if "example_text" in st.session_state:
                input_text = st.session_state.pop("example_text")
                st.rerun()

            # Выбор модели прямо в блоке
            model_choice = st.selectbox(
                "Модель:", model_names, key="single_model"
            ) if model_names else None

            clicked = st.button("🔍 Определить тональность",
                                use_container_width=True, type="primary")

        with col_out:
            st.markdown("**Результат:**")
            result_placeholder = st.empty()

            if clicked:
                if not input_text.strip():
                    result_placeholder.warning("Введите текст для анализа")
                elif not model_choice:
                    result_placeholder.error("Модели не загружены")
                else:
                    with st.spinner("Анализирую..."):
                        res, status = api_post("predict", {
                            "model_name": model_choice,
                            "text": input_text,
                        })

                    if status == 200:
                        sentiment  = res.get("sentiment", "?")
                        confidence = res.get("confidence")
                        prob_pos   = res.get("prob_pos")
                        prob_neg   = res.get("prob_neg")
                        cleaned    = res.get("cleaned_text", "")

                        badge_cls = "pos-badge" if sentiment == "Позитив" else "neg-badge"
                        emoji     = "✅" if sentiment == "Позитив" else "❌"

                        with result_placeholder.container():
                            st.markdown(
                                f"<span class='{badge_cls}'>{emoji} {sentiment}</span>",
                                unsafe_allow_html=True,
                            )

                            if confidence is not None:
                                st.write("")
                                st.markdown(f"**Уверенность модели:** {confidence:.1%}")
                                st.progress(confidence)

                            if prob_pos is not None and prob_neg is not None:
                                st.write("")
                                st.markdown("**Распределение вероятностей:**")
                                fig = go.Figure(go.Bar(
                                    x=["Негатив", "Позитив"],
                                    y=[prob_neg, prob_pos],
                                    marker_color=["#e74c3c", "#2ecc71"],
                                    text=[f"{prob_neg:.1%}", f"{prob_pos:.1%}"],
                                    textposition="outside",
                                ))
                                fig.update_layout(
                                    height=200, margin=dict(l=0, r=0, t=10, b=0),
                                    yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                                    showlegend=False,
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            if cleaned:
                                with st.expander("🔧 Текст после предобработки"):
                                    st.code(cleaned)

                    else:
                        result_placeholder.error(
                            f"Ошибка {status}: {res.get('detail', 'Неизвестная ошибка')}"
                        )
            else:
                result_placeholder.info("Введите текст и нажмите кнопку для анализа")

    # ── Пакетный анализ ───────────────────────────────────────────────────────
    with tab_batch:
        st.markdown("Введите несколько комментариев — **каждый с новой строки**:")

        batch_text = st.text_area(
            "batch",
            placeholder="Отличный день!\nУжасный сервис.\nОчень доволен покупкой.",
            height=180,
            label_visibility="collapsed",
        )

        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            batch_model = st.selectbox("Модель:", model_names, key="batch_model") if model_names else None
        with col_b2:
            batch_btn = st.button("🔍 Анализировать всё",
                                  use_container_width=True, type="primary")

        if batch_btn:
            lines = [l.strip() for l in batch_text.split("\n") if l.strip()]
            if not lines:
                st.warning("Введите хотя бы один текст")
            elif not batch_model:
                st.error("Модели не загружены")
            else:
                with st.spinner(f"Анализирую {len(lines)} текстов..."):
                    res, status = api_post("predict/batch", {
                        "model_name": batch_model,
                        "texts": lines,
                    })

                if status == 200 and isinstance(res, list):
                    rows = []
                    for item in res:
                        if "error" in item:
                            rows.append({
                                "Текст": item.get("original_text", "?")[:60],
                                "Тональность": "⚠ Ошибка",
                                "P(Позитив)": None,
                                "P(Негатив)": None,
                                "Уверенность": None,
                            })
                        else:
                            emoji = "✅" if item["label"] == 1 else "❌"
                            rows.append({
                                "Текст": item.get("original_text", "")[:60],
                                "Тональность": f"{emoji} {item['sentiment']}",
                                "P(Позитив)": item.get("prob_pos"),
                                "P(Негатив)": item.get("prob_neg"),
                                "Уверенность": item.get("confidence"),
                            })

                    df_res = pd.DataFrame(rows)
                    st.dataframe(df_res, use_container_width=True, hide_index=True)

                    # Сводка
                    valid = [r for r in res if "error" not in r]
                    if valid:
                        n_pos = sum(1 for r in valid if r["label"] == 1)
                        n_neg = len(valid) - n_pos
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Всего",     len(valid))
                        c2.metric("✅ Позитив", n_pos)
                        c3.metric("❌ Негатив", n_neg)

                        fig = px.pie(
                            values=[n_pos, n_neg],
                            names=["Позитив", "Негатив"],
                            color_discrete_sequence=["#2ecc71", "#e74c3c"],
                            hole=0.4,
                        )
                        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Ошибка {status}: {res.get('detail', '')}")

# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 2: СТАТИСТИКА ДАТАСЕТА
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Статистика датасета":
    st.subheader("📊 Статистика обучающего датасета")

    if stats is None or "error" in stats:
        st.error("Статистика недоступна. Проверьте, что файлы neg.csv/pos.csv находятся в папке Resource/")
        st.stop()

    # ── Ключевые метрики ──────────────────────────────────────────────────────
    total   = stats.get("total_comments", 0)
    neg_cnt = stats.get("negative_count", 0)
    pos_cnt = stats.get("positive_count", 0)
    neg_pct = stats.get("negative_pct", 0)
    pos_pct = stats.get("positive_pct", 0)

    st.markdown("#### Общая информация")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💬 Всего комментариев",  f"{total:,}")
    c2.metric("✅ Позитивных",           f"{pos_cnt:,}", f"{pos_pct:.1f}%")
    c3.metric("❌ Негативных",           f"{neg_cnt:,}", f"{neg_pct:.1f}%")
    c4.metric("📝 Средняя длина (слова)", f"{stats.get('avg_length_words', 0):.1f}")
    c5.metric("✍️ Уник. авторов",         f"{stats.get('unique_authors', 0):,}")

    st.divider()

    # ── Граф 1: Баланс классов ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Соотношение классов")
        fig = go.Figure(go.Pie(
            labels=["Позитив", "Негатив"],
            values=[pos_cnt, neg_cnt],
            marker_colors=["#2ecc71", "#e74c3c"],
            hole=0.42,
            textinfo="percent+label",
            hoverinfo="label+value+percent",
        ))
        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Сравнение классов")
        fig = go.Figure(go.Bar(
            x=["Негатив", "Позитив"],
            y=[neg_cnt, pos_cnt],
            marker_color=["#e74c3c", "#2ecc71"],
            text=[f"{neg_cnt:,}\n({neg_pct:.1f}%)", f"{pos_cnt:,}\n({pos_pct:.1f}%)"],
            textposition="outside",
        ))
        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=30),
            yaxis=dict(title="Количество"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Граф 2: Распределение длин текстов ───────────────────────────────────
    st.markdown("#### Распределение длин комментариев")

    col3, col4 = st.columns(2)

    with col3:
        len_dist = stats.get("length_distribution", {})
        if len_dist:
            df_ld = pd.DataFrame(
                list(len_dist.items()), columns=["Слов (диапазон)", "Комментариев"]
            )
            fig = px.bar(
                df_ld, x="Слов (диапазон)", y="Комментариев",
                color="Комментариев", color_continuous_scale="Blues",
                title="Гистограмма длин (по словам)",
            )
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        wc_sample = stats.get("word_count_sample", [])
        if wc_sample:
            fig = px.histogram(
                x=wc_sample, nbins=40,
                labels={"x": "Слов в комментарии", "y": "Количество"},
                title="Плотность распределения длин",
                color_discrete_sequence=["#4e89e8"],
            )
            avg_w  = stats.get("avg_length_words", 0)
            med_w  = stats.get("median_length_words", 0)
            fig.add_vline(x=avg_w,  line_dash="dash",  line_color="red",
                          annotation_text=f"Среднее: {avg_w:.1f}")
            fig.add_vline(x=med_w,  line_dash="dot",   line_color="orange",
                          annotation_text=f"Медиана: {med_w:.1f}")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Граф 3: Средняя длина vs Количество ──────────────────────────────────
    st.markdown("#### Детальные метрики")

    c5, c6, c7 = st.columns(3)
    c5.metric("📏 Средняя длина (символы)", f"{stats.get('avg_length_chars', 0):.1f}")
    c6.metric("📏 Медианная длина (слова)",  f"{stats.get('median_length_words', 0):.1f}")
    c7.metric("❤️ Среднее лайков",           f"{stats.get('avg_likes', 0):.2f}")

    # ── Индикатор баланса ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Баланс датасета")
    balance_score = 1 - abs(pos_pct - neg_pct) / 100
    col_bal1, col_bal2 = st.columns([1, 3])
    with col_bal1:
        color = "normal" if balance_score > 0.9 else "inverse"
        st.metric("Индекс баланса", f"{balance_score:.2%}", help="1.0 = идеальный баланс")
    with col_bal2:
        st.progress(balance_score, text=f"{'✅ Датасет сбалансирован' if balance_score > 0.9 else '⚠ Незначительный дисбаланс'}")

# ═══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА 3: СПРАВКА ПО API
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📘 Справка по API":
    st.subheader("📘 Справка по API")

    # Статус сервера
    if health:
        st.success(f"API онлайн · Загружено моделей: {health.get('models_count', 0)}")
        st.markdown(
            "**Модели:** " + ", ".join(f"`{m}`" for m in health.get("models_loaded", []))
        )
        st.markdown(
            f"**Векторизатор:** `{health.get('vectorizer', 'unknown')}`  |  "
            f"**Кэш лемм:** {health.get('lemma_cache_size', 0):,}"
        )
    else:
        st.error("API недоступен")

    st.divider()

    # Описание эндпоинтов
    endpoints = [
        {
            "badge": "GET",
            "color": "#28a745",
            "path": "GET /health",
            "desc": "Проверка работоспособности сервера.",
            "params": "—",
            "example_req": "curl http://localhost:8000/health",
            "example_resp": '{"status": "ok", "models_loaded": [...], "vectorizer": "loaded"}',
        },
        {
            "badge": "GET",
            "color": "#28a745",
            "path": "GET /models",
            "desc": "Список доступных моделей классификации.",
            "params": "—",
            "example_req": "curl http://localhost:8000/models",
            "example_resp": '[{"name": "Logistic Regression"}, {"name": "Финальная (self-trained)"}]',
        },
        {
            "badge": "GET",
            "color": "#28a745",
            "path": "GET /stats/overview",
            "desc": "Статистика обучающего датасета: количество комментариев, доли классов, средние длины.",
            "params": "—",
            "example_req": "curl http://localhost:8000/stats/overview",
            "example_resp": '{"total_comments": 217440, "positive_pct": 50.76, ...}',
        },
        {
            "badge": "POST",
            "color": "#007bff",
            "path": "POST /predict",
            "desc": "Анализ тональности одного текста. Возвращает класс, вероятности и уверенность.",
            "params": "`model_name` (str) — модель из /models\n`text` (str) — текст до 5000 символов",
            "example_req": (
                'curl -X POST http://localhost:8000/predict \\\n'
                '  -H "Content-Type: application/json" \\\n'
                '  -d \'{"model_name": "Финальная (self-trained)", "text": "Отличный день!"}\''
            ),
            "example_resp": (
                '{\n'
                '  "sentiment": "Позитив",\n'
                '  "label": 1,\n'
                '  "prob_neg": 0.1123,\n'
                '  "prob_pos": 0.8877,\n'
                '  "confidence": 0.8877,\n'
                '  "model_used": "Финальная (self-trained)"\n'
                '}'
            ),
        },
        {
            "badge": "POST",
            "color": "#007bff",
            "path": "POST /predict/batch",
            "desc": "Пакетный анализ — до 100 текстов за один запрос.",
            "params": "`model_name` (str) — модель\n`texts` (list[str]) — список текстов (макс. 100)",
            "example_req": (
                'curl -X POST http://localhost:8000/predict/batch \\\n'
                '  -H "Content-Type: application/json" \\\n'
                '  -d \'{"model_name": "Logistic Regression", "texts": ["Хорошо!", "Плохо."]}\''
            ),
            "example_resp": (
                '[\n'
                '  {"sentiment": "Позитив", "label": 1, "confidence": 0.91, ...},\n'
                '  {"sentiment": "Негатив", "label": 0, "confidence": 0.87, ...}\n'
                ']'
            ),
        },
        {
            "badge": "GET",
            "color": "#28a745",
            "path": "GET /docs",
            "desc": "Интерактивная Swagger-документация — тестируйте запросы прямо в браузере.",
            "params": "—",
            "example_req": "Откройте http://localhost:8000/docs в браузере",
            "example_resp": "Swagger UI (интерактивный интерфейс)",
        },
    ]

    for ep in endpoints:
        color = ep["color"]
        with st.expander(f"**{ep['path']}**"):
            st.markdown(f"**Описание:** {ep['desc']}")
            st.markdown(f"**Параметры:**\n{ep['params']}")
            st.markdown("**Пример запроса:**")
            st.code(ep["example_req"], language="bash")
            st.markdown("**Пример ответа:**")
            st.code(ep["example_resp"], language="json")

    st.divider()

    # Коды ошибок
    st.markdown("#### Коды HTTP-ответов")
    df_codes = pd.DataFrame([
        {"Код": "200 OK",              "Описание": "Успешный запрос"},
        {"Код": "400 Bad Request",     "Описание": "Пустой текст после предобработки или некорректный запрос"},
        {"Код": "404 Not Found",       "Описание": "Указанная модель не найдена"},
        {"Код": "503 Service Unavail.","Описание": "Модели или векторизатор не загружены"},
        {"Код": "500 Server Error",    "Описание": "Внутренняя ошибка сервера"},
    ])
    st.dataframe(df_codes, use_container_width=True, hide_index=True)

    st.divider()

    # Быстрый старт
    st.markdown("#### 🚀 Быстрый старт")
    st.code("""
# 1. Установите зависимости
pip install fastapi uvicorn pymorphy3 nltk scikit-learn joblib pandas streamlit plotly requests

# 2. Запустите API (в отдельном терминале)
python Api.py

# 3. Запустите приложение
streamlit run streamlit_app.py

# 4. Откройте браузер
#    Приложение:  http://localhost:8501
#    API / Docs:  http://localhost:8000/docs
    """, language="bash")

    # Структура файлов
    st.markdown("#### 📁 Ожидаемая структура файлов")
    st.code("""
project/
├── Api.py                      ← FastAPI-сервер
├── streamlit_app.py            ← Streamlit-приложение
├── Models/
│   ├── tfidf_vectorizer.pkl    ← TF-IDF векторизатор
│   ├── lemma_cache.pkl         ← Кэш лемматизации
│   ├── model_logistic_regression.pkl
│   ├── model_naive_bayes.pkl
│   ├── model_linear_svc.pkl
│   └── final_model.pkl         ← Финальная модель (self-trained)
└── Resource/
    ├── neg.csv                 ← Негативные твиты
    └── pos.csv                 ← Позитивные твиты
    """, language="")
