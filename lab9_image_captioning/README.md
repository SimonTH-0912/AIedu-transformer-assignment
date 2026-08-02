# Lab 9 Streamlit 앱

BLIP base로 영어 이미지 캡션을 생성하고 MarianMT로 한국어 번역을 제공합니다. COCO 데이터셋 이미지에서는 제공된 정답 캡션과 비교할 수 있습니다.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Community Cloud의 Main file path는 `lab9_image_captioning/app.py`입니다. 첫 실행 시 `Salesforce/blip-image-captioning-base`와 한국어 번역 모델을 내려받으므로 시간이 걸립니다. 번역 모델 로딩에 실패해도 영어 캡션은 표시됩니다. JPG/JPEG/PNG를 최대 10개, 파일당 10 MB까지 지원합니다.
