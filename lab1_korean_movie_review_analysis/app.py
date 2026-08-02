from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core import InputValidationError, clean_sentences, count_tokens, decode_text, top_tokens


DEFAULT_DATA = BASE_DIR / "data" / "input.txt"
DEFAULT_STOPWORDS = "영화 정말 너무 그냥 진짜 보고 하는 있는 없는"

st.set_page_config(page_title="Lab 1 · 한국어 리뷰 분석", page_icon="🎬", layout="wide")


@st.cache_resource
def get_tagger():
    from konlpy.tag import Okt

    return Okt()


def draw_bar(rows):
    labels = [name for name, _ in rows][::-1]
    values = [value for _, value in rows][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.28)))
    ax.barh(labels, values, color="#5B7CFA")
    ax.set_xlabel("빈도")
    ax.set_title("상위 토큰")
    fig.tight_layout()
    return fig


def draw_wordcloud(counter):
    font_candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((str(path) for path in font_candidates if path.exists()), None)
    cloud = WordCloud(width=1000, height=500, background_color="white", font_path=font_path)
    cloud.generate_from_frequencies(counter)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    return fig


st.title("🎬 한국어 영화 리뷰 분석 실험실")
st.caption("정제 → 형태소 분석 → 필터링 → 빈도 시각화의 전체 흐름을 직접 실험합니다.")

with st.expander("학습 목표와 처리 원리", expanded=True):
    st.markdown("""
    1. 정규표현식으로 한글 문장만 남깁니다.
    2. Okt가 형태소와 품사를 판별합니다.
    3. 원하는 품사와 불용어 조건을 적용하고 빈도를 집계합니다.
    4. 같은 빈도 데이터를 막대그래프와 워드클라우드로 비교합니다.
    """)

source = st.radio("데이터", ["내장 영화 리뷰", "TXT 업로드"], horizontal=True)
uploaded = st.file_uploader("UTF-8 TXT 파일", type=["txt"], disabled=source == "내장 영화 리뷰")

with st.sidebar:
    st.header("실험 설정")
    allowed_pos = set(st.multiselect("포함 품사", ["Noun", "Verb", "Adjective", "Adverb"], ["Noun", "Verb", "Adjective"]))
    stopword_text = st.text_area("불용어 (공백 구분)", DEFAULT_STOPWORDS)
    min_length = st.slider("최소 토큰 길이", 1, 6, 2)
    min_freq = st.slider("최소 빈도", 1, 20, 2)
    top_n = st.slider("상위 토큰 수", 5, 50, 20)

try:
    if source == "내장 영화 리뷰":
        raw = DEFAULT_DATA.read_bytes()
    elif uploaded is not None:
        raw = uploaded.getvalue()
    else:
        raw = b""
    text = decode_text(raw) if raw else ""
    if text:
        st.subheader("1. 입력 미리보기")
        st.text_area("원문", text[:5000], height=160, disabled=True)
        sentences = clean_sentences(text)
        st.info(f"{len(sentences):,}개 문장, {len(text):,}자를 분석합니다.")

        if st.button("분석 실행", type="primary", width="stretch"):
            if not allowed_pos:
                raise InputValidationError("품사를 하나 이상 선택해 주세요.")
            with st.spinner("Okt 형태소 분석 중…"):
                counts = count_tokens(sentences, get_tagger(), set(stopword_text.split()), allowed_pos, min_length, min_freq)
            if not counts:
                raise InputValidationError("현재 필터 조건을 만족하는 토큰이 없습니다.")
            rows = top_tokens(counts, top_n)
            frame = pd.DataFrame(rows, columns=["토큰", "빈도"])
            st.subheader("2. 분석 결과")
            tab1, tab2, tab3, tab4 = st.tabs(["정제 문장", "빈도표", "막대그래프", "워드클라우드"])
            tab1.dataframe(pd.DataFrame({"정제 문장": sentences[:200]}), width="stretch")
            tab2.dataframe(frame, width="stretch", hide_index=True)
            tab2.download_button("CSV 내려받기", frame.to_csv(index=False).encode("utf-8-sig"), "top_tokens.csv", "text/csv")
            tab3.pyplot(draw_bar(rows))
            tab4.pyplot(draw_wordcloud(counts))
            st.success("최소 빈도나 품사를 바꿔 두 시각화가 어떻게 달라지는지 비교해 보세요.")
    else:
        st.info("분석할 TXT 파일을 업로드해 주세요.")
except (InputValidationError, OSError) as exc:
    st.error(str(exc))
