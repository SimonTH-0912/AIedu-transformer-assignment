from pathlib import Path
import sys

import streamlit as st

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from core import (GenerationOptions, ImageItem, InputValidationError, build_caption_result,
                  generate_caption, load_blip, load_reference_captions, load_translator,
                  translate_caption, validate_images)

COCO_DIR = BASE / "coco_images"
REF_PATH = BASE / "korean_image_captioning_dataset" / "MSCOCO_train_val_Korean.json"
st.set_page_config(page_title="Lab 9 · 이미지 캡셔닝", page_icon="🖼️", layout="wide")


@st.cache_resource(show_spinner="BLIP 모델을 처음 내려받는 중입니다…")
def cached_blip():
    return load_blip()


@st.cache_resource(show_spinner="영→한 번역 모델을 내려받는 중입니다…")
def cached_translator():
    return load_translator()


@st.cache_data
def cached_references():
    return load_reference_captions(REF_PATH)


st.title("🖼️ Transformer 이미지 캡셔닝 실험실")
st.caption("ViT 기반 이미지 인코더와 언어 디코더가 이미지를 문장으로 바꾸는 과정을 관찰합니다.")
with st.expander("BLIP와 CLIP의 차이", expanded=True):
    st.markdown("**CLIP**은 이미지와 문장의 유사도를 비교하고, **BLIP**은 이미지 특징을 조건으로 다음 단어를 반복 생성합니다. Beam 수를 늘리면 여러 문장 후보를 비교하지만 추론 시간도 늘어납니다.")

source = st.radio("이미지", ["내장 COCO 이미지", "이미지 업로드"], horizontal=True)
if source == "내장 COCO 이미지":
    available = sorted(COCO_DIR.glob("*.jpg"))
    selected_names = st.multiselect("최대 10개 선택", [p.name for p in available], [p.name for p in available[:1]], max_selections=10)
    selected = [COCO_DIR / name for name in selected_names]
    items = [ImageItem(path.name, path.read_bytes()) for path in selected]
else:
    uploads = st.file_uploader("최대 10개 이미지", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    items = [ImageItem(file.name, file.getvalue()) for file in uploads]

with st.sidebar:
    st.header("생성 설정")
    max_length = st.slider("최대 토큰 길이", 5, 60, 30)
    num_beams = st.slider("Beam 수", 1, 8, 3)
    prompt = st.text_input("조건 프롬프트 (선택)", "")
    translate = st.checkbox("한국어로 번역", True)

try:
    if not items:
        st.info("이미지를 선택하거나 업로드해 주세요."); st.stop()
    images = validate_images(items)
    st.subheader("1. 입력 이미지")
    preview_cols = st.columns(min(5, len(items)))
    for idx, (item, image) in enumerate(zip(items, images)):
        preview_cols[idx % len(preview_cols)].image(image, caption=item.name, width="stretch")

    if st.button("캡션 생성", type="primary", width="stretch"):
        options = GenerationOptions(max_length, num_beams)
        processor, model = cached_blip()
        refs = cached_references() if REF_PATH.exists() else {}
        results = []
        progress = st.progress(0)
        for idx, (item, image) in enumerate(zip(items, images), 1):
            english = generate_caption(image, processor, model, options, prompt)
            korean = warning = None
            if translate:
                try:
                    tokenizer, mt_model = cached_translator()
                    korean = translate_caption(english, tokenizer, mt_model)
                except Exception as exc:
                    warning = f"한국어 번역을 완료하지 못했습니다: {exc}"
            results.append(build_caption_result(item.name, english, korean, warning, refs.get(item.name)))
            progress.progress(idx / len(items), f"{idx}/{len(items)} 처리")

        st.subheader("2. 생성 결과")
        for result, image in zip(results, images):
            with st.container(border=True):
                left, right = st.columns([1, 2])
                left.image(image, caption=result.filename, width="stretch")
                right.markdown(f"**영어 생성:** {result.english}")
                if result.korean: right.markdown(f"**한국어 번역:** {result.korean}")
                if result.translation_warning: right.warning(result.translation_warning)
                if result.reference:
                    with right.expander("데이터셋 정답 캡션과 비교"):
                        st.write("영어", result.reference.english); st.write("한국어", result.reference.korean)
        st.info("Beam 수와 조건 프롬프트를 바꿔 문장의 구체성과 생성 시간이 어떻게 달라지는지 비교해 보세요.")
        st.caption("`blip-image-captioning-large`는 품질이 높지만 Cloud 메모리 사용량이 커서 확장 실습으로만 권장합니다.")
except (InputValidationError, OSError, RuntimeError) as exc:
    st.error(str(exc))
