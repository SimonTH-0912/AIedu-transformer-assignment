from pathlib import Path
import sys

import streamlit as st

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from core import (ImageItem, InputValidationError, encode_images, encode_text,
                  load_clip, rank_similar, validate_image)

IMAGE_DIR = BASE / "fashion_images"
st.set_page_config(page_title="Lab 6 · CLIP 패션 검색", page_icon="👗", layout="wide")


@st.cache_resource(show_spinner="CLIP 모델을 처음 내려받는 중입니다…")
def cached_model():
    return load_clip()


@st.cache_data(show_spinner="패션 이미지 임베딩 생성 중…")
def cached_embeddings(payloads):
    processor, model = cached_model()
    images = [validate_image(data, name) for name, data in payloads]
    return encode_images(images, processor, model)


st.title("👗 CLIP 멀티모달 패션 검색")
st.caption("텍스트와 이미지가 같은 512차원 의미 공간에 놓이는 과정을 실험합니다.")
with st.expander("CLIP 검색 원리", expanded=True):
    st.markdown("텍스트 인코더와 이미지 인코더가 각각 벡터를 만들고, 정규화된 벡터의 **코사인 유사도**가 큰 이미지를 찾습니다. 분류 라벨 없이도 ‘우아한 검은 드레스’ 같은 문장으로 검색할 수 있습니다.")

source = st.radio("이미지 모음", ["내장 패션 이미지", "이미지 업로드"], horizontal=True)
uploads = st.file_uploader("최대 30개 이미지", type=["jpg", "jpeg", "png"], accept_multiple_files=True, disabled=source == "내장 패션 이미지")

try:
    if source == "내장 패션 이미지":
        paths = sorted(path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        items = [ImageItem(path.name, path.read_bytes()) for path in paths]
    else:
        if len(uploads) > 30:
            raise InputValidationError("이미지는 최대 30개까지 업로드할 수 있습니다.")
        items = [ImageItem(file.name, file.getvalue()) for file in uploads]
    if not items:
        st.info("검색할 이미지를 업로드해 주세요."); st.stop()

    images = [validate_image(item.data, item.name) for item in items]
    st.subheader("1. 이미지 컬렉션")
    cols = st.columns(min(6, len(items)))
    for idx, (item, image) in enumerate(zip(items, images)):
        cols[idx % len(cols)].image(image, caption=item.name, width="stretch")

    payloads = tuple((item.name, item.data) for item in items)
    collection_key = tuple((item.name, len(item.data)) for item in items)
    if st.button("CLIP 검색 인덱스 만들기", type="primary", width="stretch"):
        st.session_state["clip_vectors"] = cached_embeddings(payloads)
        st.session_state["clip_collection_key"] = collection_key
    vectors = st.session_state.get("clip_vectors") if st.session_state.get("clip_collection_key") == collection_key else None
    if vectors is None:
        st.info("검색 전에 버튼을 눌러 이미지들을 CLIP 벡터로 변환해 주세요.")
        st.stop()
    st.info(f"{len(items)}개 이미지를 {vectors.shape[1]}차원 단위 벡터로 변환했습니다.")

    with st.sidebar:
        st.header("검색 설정")
        method = st.radio("쿼리 방식", ["텍스트", "이미지"])
        top_n = st.slider("결과 수", 1, len(items), min(5, len(items)))

    processor, model = cached_model()
    if method == "텍스트":
        query_text = st.text_input("찾고 싶은 스타일", "깔끔한 흰색 스니커즈")
        ready = st.button("텍스트로 검색", type="primary")
        query_vector = encode_text(query_text, processor, model) if ready else None
    else:
        query_file = st.file_uploader("쿼리 이미지", type=["jpg", "jpeg", "png"], key="query")
        ready = st.button("이미지로 검색", type="primary") and query_file is not None
        query_image = validate_image(query_file.getvalue(), query_file.name) if ready else None
        query_vector = encode_images([query_image], processor, model)[0] if ready else None

    if query_vector is not None:
        st.subheader("2. 유사도 검색 결과")
        results = rank_similar(query_vector, vectors, top_n)
        cols = st.columns(min(5, len(results)))
        for slot, result in enumerate(results):
            item, image = items[result.index], images[result.index]
            cols[slot % len(cols)].image(image, caption=f"{item.name}\n유사도 {result.score:.3f}", width="stretch")
        st.caption("점수가 높을수록 CLIP의 공유 임베딩 공간에서 쿼리와 가까운 이미지입니다.")
except (InputValidationError, OSError, RuntimeError) as exc:
    st.error(str(exc))
    st.caption("모델 다운로드 오류라면 잠시 후 재시도하거나 Streamlit Cloud 앱을 재부팅해 주세요.")
