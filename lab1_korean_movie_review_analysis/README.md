# Lab 1 Streamlit 앱

한국어 영화 리뷰의 전처리, Okt 형태소 분석, 빈도표, 막대그래프와 워드클라우드를 단계별로 실험합니다.

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

저장소를 연결한 뒤 Main file path를 `lab1_korean_movie_review_analysis/app.py`로 지정합니다. `packages.txt`가 Java 런타임과 나눔 한글 폰트를 설치하며 별도 비밀 키는 필요 없습니다. KoNLPy/Okt가 JVM을 사용하므로 첫 분석은 시간이 걸릴 수 있습니다.

UTF-8 TXT는 4 MB까지 지원하며, 이 범위 안에서는 파일 전체를 분석합니다. 내장 `data/input.txt`도 바로 사용할 수 있습니다.
