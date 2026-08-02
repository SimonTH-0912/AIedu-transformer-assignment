from pathlib import Path
import sys

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from core import (InputValidationError, beam_translate, build_vocabularies, greedy_translate,
                  load_parallel_csv, load_parallel_text, tokenize_english, tokenize_korean,
                  train_model, training_fingerprint)

st.set_page_config(page_title="Lab 2 · Seq2Seq 번역", page_icon="🌐", layout="wide")
st.title("🌐 Seq2Seq 한→영 번역 실험실")
st.caption("병렬 말뭉치로 작은 Attention-GRU 모델을 직접 학습하고 디코딩 방법을 비교합니다.")

with st.expander("모델 흐름", expanded=True):
    st.markdown("한국어 문장 → **Encoder GRU** → 문맥과 Attention → **Decoder GRU** → 영어 토큰. 이 앱은 교육용 소형 모델이므로 번역 품질보다 학습 흐름 관찰에 초점을 둡니다.")

source = st.radio("데이터", ["내장 병렬 TXT", "TXT 두 파일", "CSV"], horizontal=True)
try:
    if source == "내장 병렬 TXT":
        pairs = load_parallel_text((BASE / "data/train_kor.txt").read_bytes(), (BASE / "data/train_eng.txt").read_bytes())
    elif source == "TXT 두 파일":
        kor = st.file_uploader("한국어 TXT", type="txt", key="kor")
        eng = st.file_uploader("영어 TXT", type="txt", key="eng")
        pairs = load_parallel_text(kor.getvalue(), eng.getvalue()) if kor and eng else []
    else:
        csv_file = st.file_uploader("korean, english 열이 있는 CSV", type="csv")
        pairs = load_parallel_csv(csv_file.getvalue()) if csv_file else []

    if not pairs:
        st.info("병렬 데이터를 선택하거나 업로드해 주세요.")
        st.stop()

    with st.sidebar:
        st.header("학습 설정")
        max_samples = min(5000, len(pairs))
        sample_count = st.slider("학습 문장쌍", 1, max_samples, min(300, max_samples))
        epochs = st.slider("에포크", 1, 20, 3)
        learning_rate = st.select_slider("학습률", [0.0005, 0.001, 0.003, 0.005, 0.01], 0.003)
        hidden_dim = st.select_slider("은닉 차원", [32, 64, 128], 64)
        seed = st.number_input("난수 시드", 0, 9999, 7)

    active_pairs = pairs[:sample_count]
    st.subheader("1. 데이터와 토큰")
    st.dataframe(pd.DataFrame(active_pairs[:20], columns=["한국어", "영어"]), width="stretch", hide_index=True)
    src_vocab, tgt_vocab = build_vocabularies(active_pairs)
    sample_src, sample_tgt = active_pairs[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("문장쌍", len(active_pairs)); c2.metric("한국어 어휘", len(src_vocab)); c3.metric("영어 어휘", len(tgt_vocab))
    with st.expander("토큰화 예시"):
        st.write("한국어", tokenize_korean(sample_src)); st.write("영어", tokenize_english(sample_tgt))

    settings = {"samples": sample_count, "epochs": epochs, "lr": learning_rate, "hidden": hidden_dim, "seed": int(seed)}
    fingerprint = training_fingerprint(active_pairs, settings)
    if st.button("모델 학습", type="primary", width="stretch"):
        progress = st.progress(0, "학습 준비")
        chart = st.empty()
        observed = []
        def update(epoch, total, loss):
            observed.append(loss); progress.progress(epoch / total, f"에포크 {epoch}/{total} · loss {loss:.3f}"); chart.line_chart(observed)
        bundle, losses = train_model(active_pairs, epochs, learning_rate, hidden_dim=hidden_dim, seed=int(seed), progress_callback=update)
        st.session_state["bundle"] = bundle; st.session_state["fingerprint"] = fingerprint; st.session_state["losses"] = losses
        st.success("현재 브라우저 세션에 모델을 저장했습니다.")

    if "bundle" in st.session_state:
        if st.session_state.get("fingerprint") != fingerprint:
            st.warning("데이터나 학습 설정이 변경되었습니다. 현재 설정으로 다시 학습해 주세요.")
        else:
            st.subheader("2. 직접 번역")
            sentence = st.text_input("한국어 문장", active_pairs[0][0])
            beam_width = st.slider("Beam 폭", 1, 8, 3)
            if st.button("번역 비교") and sentence.strip():
                greedy = " ".join(greedy_translate(sentence, st.session_state.bundle))
                beam = " ".join(beam_translate(sentence, st.session_state.bundle, beam_width))
                a, b = st.columns(2); a.info(f"Greedy\n\n{greedy or '(빈 결과)'}"); b.success(f"Beam Search\n\n{beam or '(빈 결과)'}")
                st.caption("Greedy는 매 단계 최댓값 하나를, Beam Search는 여러 후보 경로를 유지합니다.")
except (InputValidationError, OSError) as exc:
    st.error(str(exc))
