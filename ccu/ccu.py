import os, re, math, random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, List, Any
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline
)
from tqdm import tqdm
from collections import Counter

# -------------------------------------------------------------------
# 기본 설정
# -------------------------------------------------------------------
Image.MAX_IMAGE_PIXELS = None
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------------------------------------------------
# 데이터 로드
# -------------------------------------------------------------------
train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")
train_df = train_df.sample(n=200, random_state=SEED).reset_index(drop=True)

# -------------------------------------------------------------------
#  영어 번역 파이프라인 (한국어 → 영어)
# -------------------------------------------------------------------
print("🔤 Translating Korean questions to English...")
translator = pipeline("translation", model="facebook/m2m100_418M")
test_df["question_en"] = [
    translator(q, src_lang="ko", tgt_lang="en")[0]["translation_text"]
    for q in tqdm(test_df["question"].tolist())
]

# -------------------------------------------------------------------
# 모델 후보 5개
# -------------------------------------------------------------------
MODEL_IDS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-flan-t5-xl",
    "google/paligemma-3b-ft-ocrvqa-448",
    "deepseek-ai/deepseek-vl-7b-chat",
]

IMAGE_SIZE = 384
MAX_NEW_TOKENS = 8

# -------------------------------------------------------------------
# 양자화 설정
# -------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# -------------------------------------------------------------------
# 응답 파서
# -------------------------------------------------------------------
def extract_choice(text: str) -> str:
    text = text.strip().lower()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "a"
    last = lines[-1]
    if last in ["a", "b", "c", "d"]:
        return last
    tokens = last.split()
    for tok in tokens:
        if tok in ["a", "b", "c", "d"]:
            return tok
    return "error"

# -------------------------------------------------------------------
# 개선된 프롬프트 구성
# -------------------------------------------------------------------
SYSTEM_INSTRUCT = (
    "You are a world-class visual reasoning assistant. "
    "You will be given an image and a multiple-choice question. "
    "Carefully analyze the image and reason step by step to select the most accurate answer. "
    "Respond with only one lowercase letter (a, b, c, or d)."
)

# Few-shot 예시
FEW_SHOT_EXAMPLES = """
Example 1:
Image: [A man holding an umbrella in the rain]
Question: What is the weather like?
(a) Sunny (b) Rainy (c) Snowy (d) Cloudy
Answer: b

Example 2:
Image: [A child playing with a ball in a park]
Question: Where is the child?
(a) Beach (b) Park (c) School (d) Restaurant
Answer: b

Now, analyze the next image carefully.
"""

def prompt_cot(question_en, a, b, c, d):
    return (
        f"Question: {question_en}\n"
        f"Choices:\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "Think step by step and reason logically before deciding. "
        "Answer only with one lowercase letter (a, b, c, or d)."
    )

def prompt_elimination(question_en, a, b, c, d):
    return (
        f"Question: {question_en}\n"
        f"Choices:\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "Eliminate clearly incorrect options one by one, then select the final best answer. "
        "Respond with only the final letter."
    )

def prompt_fewshot(question_en, a, b, c, d):
    return (
        FEW_SHOT_EXAMPLES +
        f"\nQuestion: {question_en}\n"
        f"Choices:\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "Reason step by step and output only one lowercase letter (a, b, c, or d)."
    )

PROMPT_FUNCS = [prompt_cot, prompt_elimination, prompt_fewshot]

# -------------------------------------------------------------------
# 모델 로드
# -------------------------------------------------------------------
models, processors = [], []
print("\n===== Loading 5 Models for Ensemble =====")
for MODEL_ID in MODEL_IDS:
    print(f"Loading: {MODEL_ID}")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        processors.append(processor)
        models.append(model)
    except Exception as e:
        print(f"[Error] {MODEL_ID} load failed:", e)

print(f"\n Loaded {len(models)} models successfully.\n")

# -------------------------------------------------------------------
# 추론 (모델 × 프롬프트 앙상블)
# -------------------------------------------------------------------
final_preds = []
all_model_preds = {m: [] for m in MODEL_IDS}

for i in tqdm(range(len(test_df)), desc="Inference Ensemble", unit="sample"):
    row = test_df.iloc[i]
    img = Image.open(row["path"]).convert("RGB")

    votes = []  # 모델+프롬프트 조합별 투표
    for MODEL_ID, model, processor in zip(MODEL_IDS, models, processors):
        for build_prompt in PROMPT_FUNCS:
            user_text = build_prompt(row["question_en"], row["a"], row["b"], row["c"], row["d"])

            try:
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_text}
                    ]}
                ]

                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    # 💡 PaliGemma 계열만 <image> 토큰 자동 추가
                    if "paligemma" in MODEL_ID.lower() and "<image>" not in text:
                        text = "<image>\n" + text

                    inputs = processor(
                        text=[text],
                        images=[img],
                        return_tensors="pt"
                    ).to(device)

                    # 💡 모델 dtype과 입력 dtype 자동 동기화 (fp16 호환)
                    model_dtype = getattr(model, "dtype", torch.float32)
                    if model_dtype == torch.float16:
                        inputs = {k: v.to(dtype=torch.float16) for k, v in inputs.items()}

                except Exception as e:
                    # 💡 fallback (BLIP2, DeepSeek 등 구조가 다른 모델용)
                    print(f"[Warning] Processor chat_template failed for {MODEL_ID}: {e}")
                    inputs = processor(images=img, text=user_text, return_tensors="pt").to(device)

                    model_dtype = getattr(model, "dtype", torch.float32)
                    if model_dtype == torch.float16:
                        inputs = {k: v.to(dtype=torch.float16) for k, v in inputs.items()}

                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=3,
                        do_sample=False,
                        eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None,
                    )

                output_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
                choice = extract_choice(output_text)
            except Exception as e:
                print(f"[Error] {MODEL_ID} 추론 실패: {e}")
                choice = "a"

            votes.append(choice)

        # 모델별로 가장 많이 나온 선택을 기록
        majority = Counter(votes).most_common(1)[0][0]
        all_model_preds[MODEL_ID].append(majority)

    # 전체 투표 (모든 모델 × 프롬프트)
    final_answer = Counter(votes).most_common(1)[0][0]
    final_preds.append(final_answer)

# -------------------------------------------------------------------
# 1. 제출용 CSV (채점용)
# -------------------------------------------------------------------
submission = pd.DataFrame({
    "id": test_df["id"],
    "answer": final_preds
})
submission.to_csv("./submission_ensemble_v2.csv", index=False)
print("\nSaved ./submission_ensemble_v2.csv")

# -------------------------------------------------------------------
# 2. 모델별 결과 CSV
# -------------------------------------------------------------------
model_result_df = pd.DataFrame({"id": test_df["id"]})
for MODEL_ID in MODEL_IDS:
    safe_name = MODEL_ID.replace("/", "_")
    model_result_df[safe_name] = all_model_preds[MODEL_ID]
model_result_df.to_csv("./model_predictions_v2.csv", index=False)
print("Saved ./model_predictions_v2.csv")

# -------------------------------------------------------------------
# 3. 정확도 계산
# -------------------------------------------------------------------
if "answer" in test_df.columns:
    correct = sum(test_df["answer"] == submission["answer"])
    total = len(test_df)
    acc = correct / total * 100
    print(f"\nEnsemble Accuracy: {acc:.2f}% ({correct}/{total})")
else:
    print("\n'test.csv'에 'answer' 컬럼이 없어 정확도 계산 생략됨.")
