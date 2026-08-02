# Lab 2 Streamlit 앱

소규모 한영 병렬 말뭉치로 Attention-GRU Seq2Seq 모델을 직접 학습하고 Greedy/Beam Search 결과를 비교합니다.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Community Cloud의 Main file path는 `lab2_seq2seq_translation_modeling/app.py`입니다. CSV 업로드 시 열 이름은 `korean`, `english`여야 합니다. 파일당 5 MB, 5,000 문장쌍, 20 에포크로 제한되며 CPU에서는 기본 300쌍·3 에포크부터 시작하는 것을 권장합니다. 학습 모델은 현재 브라우저 세션에만 유지됩니다.
