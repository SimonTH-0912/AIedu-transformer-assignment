# Lab 6 Streamlit 앱

OpenAI CLIP 체크포인트를 이용해 텍스트 또는 이미지로 패션 이미지를 검색합니다.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Community Cloud에서는 Main file path를 `lab6_clip_multimodal_fashion_search/app.py`로 지정합니다. 앱은 Hugging Face의 `openai/clip-vit-base-patch16`을 첫 실행에 내려받아 캐시합니다. 비밀 키는 필요 없습니다. JPG/JPEG/PNG를 최대 30개, 파일당 10 MB까지 지원합니다.
