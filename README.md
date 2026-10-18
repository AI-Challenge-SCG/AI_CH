
## 🧮 Model Optimization & Mathematical Approach

### 1. 4-bit Quantization & LoRA

A100 GPU 메모리 사용량을 줄이기 위해 Qwen 기반 VLM에
4-bit NF4 양자화와 LoRA 파인튜닝을 적용했습니다.

#### 4-bit Quantization

파라미터 수가 N이고, 파라미터당 비트 수가 b일 때
가중치 저장에 필요한 이론적 메모리는 다음과 같습니다.

$$
M = \frac{N \times b}{8}
$$

7B 모델 기준 가중치 메모리를 비교하면 다음과 같습니다.

| Precision | Memory |
|---|---:|
| FP32 | 28 GB |
| FP16 | 14 GB |
| 4-bit | 3.5 GB |

FP16 대비 가중치 저장 공간을 이론적으로 약 75% 절감할 수 있습니다.
실제 GPU 메모리 사용량은 양자화 메타데이터와 Activation 등에 따라 달라집니다.

#### LoRA Fine-Tuning

기존 가중치 W는 고정하고 저차원 행렬 A, B를 학습합니다.

$$
W = W_0 + \frac{\alpha}{r}BA
$$

학습 파라미터 수는 다음과 같이 감소합니다.

$$
P_{Full}=d_{in}\times d_{out}
$$

$$
P_{LoRA}=r(d_{in}+d_{out})
$$

본 프로젝트에서는 `r=8`, `alpha=16`을 사용했습니다.

예를 들어 4096×4096 Linear Layer에서는
16,777,216개 대신 65,536개 파라미터를 학습합니다.

이는 해당 Layer의 약 **0.39%**에 해당합니다.

---

### 2. A/B/C/D Prediction Probability Extraction

Qwen VLM은 일반적인 분류 모델과 달리
다음 토큰을 생성하는 Autoregressive 모델입니다.

따라서 각 선택지에 대한 확률을 얻기 위해
출력 토큰의 Logit을 활용했습니다.

#### Step 1. Prompt Construction

이미지와 질문을 함께 입력하고,
정답을 a, b, c, d 중 하나로 제한했습니다.

```text
[Image]

Question: What color is the car?

(a) Red
(b) Blue
(c) Green
(d) Yellow

Answer using exactly one letter: a, b, c, or d.
```

#### Step 2. Token Logit Extraction

모델이 다음 토큰을 생성하기 직전의 Logit을 얻습니다.

$$
Z = f_{\theta}(Image, Question)
$$

전체 Vocabulary에 대한 Logit 중
선택지 a, b, c, d에 해당하는 값만 추출합니다.

```python
import torch

# 모델의 다음 토큰 Logit
with torch.no_grad():
    outputs = model(**inputs)

next_logits = outputs.logits[0, -1, :]

# 각 선택지의 토큰 ID
choices = ["a", "b", "c", "d"]

token_ids = [
    processor.tokenizer.encode(
        c, add_special_tokens=False
    )[0]
    for c in choices
]

# A/B/C/D에 해당하는 Logit
choice_logits = next_logits[token_ids]
```

이 코드는 각 선택지가 단일 토큰이고,
입력이 정답 생성 직전 위치에서 끝나는 경우의 예시입니다.

실제 구현 시에는 공백이 포함된 토큰 여부와
Chat Template, 토큰 분할을 함께 확인해야 합니다.

#### Step 3. Softmax Normalization

추출한 4개 Logit에 Softmax를 적용하여
선택지별 정규화 확률을 계산합니다.

$$
P(y=k\mid x)=
\frac{e^{z_k}}
{\sum_{j\in\{a,b,c,d\}}e^{z_j}}
$$

```python
choice_probs = torch.softmax(
    choice_logits.float(), dim=-1
)

# 예측 확률과 최종 선택지
probabilities = choice_probs.cpu().tolist()
prediction = choices[choice_probs.argmax().item()]
```

예시 결과:

| Choice | Logit | Probability |
|---|---:|---:|
| a | 1.0 | 0.10 |
| b | 3.0 | 0.71 |
| c | 1.5 | 0.16 |
| d | -0.5 | 0.02 |

※ 예시 확률은 반올림한 값입니다.

**최종 예측: b**

이 확률은 전체 Vocabulary가 아니라
4개 선택지만을 대상으로 정규화한 조건부 확률입니다.

---

### 3. Logistic Regression Stacking Ensemble

단일 VLM의 예측 오류를 보완하기 위해
3개 VLM의 선택지별 확률을 결합했습니다.

#### Feature Construction

각 모델은 4개 선택지의 확률을 출력합니다.

$$
P_m=[p_a,p_b,p_c,p_d]
$$

이를 연결해 12차원 Feature Vector를 구성합니다.

$$
X=[P_1,P_2,P_3]\in\mathbb{R}^{12}
$$

예시:

```python
model_a = [0.10, 0.65, 0.15, 0.10]
model_b = [0.15, 0.55, 0.20, 0.10]
model_c = [0.40, 0.25, 0.20, 0.15]

features = model_a + model_b + model_c
# 12-dimensional feature
```

#### Meta-Model Training

Logistic Regression을 활용해 모델별 예측 확률의
조합을 학습했습니다.

클래스별 점수:

$$
s_k=w_k^TX+b_k
$$

최종 클래스 확률:

$$
P(y=k\mid X)=
\frac{e^{s_k}}{\sum_{j=1}^{4}e^{s_j}}
$$

최종 예측:

$$
\hat{y}=\arg\max_k P(y=k\mid X)
$$

학습 시 Cross-Entropy Loss를 최소화합니다.

$$
L=-\frac{1}{N}
\sum_{i=1}^{N}\log P(y_i\mid X_i)
$$

단순 다수결과 달리 모델별 확률 분포를
함께 고려하여 최종 정답을 결정합니다.

Meta-Model은 데이터 누수를 방지하기 위해
평가 데이터와 분리된 예측 결과로 학습해야 합니다.

---

### 4. Evaluation

모델의 성능은 Accuracy로 평가했습니다.

$$
Accuracy=\frac{N_{correct}}{N_{total}}
$$

| Metric | Result |
|---|---:|
| Validation Accuracy | **0.96 (96%)** |
| Kaggle Ranking | **9th / ~1,000 Teams** |
| Validation Samples | 750 |
| Test Samples | 750 |
| Ensemble Features | 12 |
| Meta-Model | Logistic Regression |

### Summary

| Technique | Purpose |
|---|---|
| 4-bit Quantization | 모델 가중치 메모리 절감 |
| LoRA | 학습 파라미터 수 감소 |
| Token Logit Extraction | A/B/C/D 선택지 확률 계산 |
| Softmax | 선택지별 확률 정규화 |
| Logistic Regression Stacking | 3개 VLM의 예측 결합 |
| Accuracy | 최종 모델 성능 평가 |
