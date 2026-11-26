"""
ISOM5160 Homework 2 — SMS Spam / Smishing Alert App
Course: ISOM5160-L3 Python Programming for Business Applications, Fall 2026 (PG ICSM)

INSTRUCTIONS (read carefully)
- Complete the marked STUDENT sections and the function bodies below.
- Keep these function names: load_model, load_test_samples, classify_text,
  compute_accuracy, main.
- Call main() at the bottom of this file (the starter already does this).
- In GitHub and on Streamlit Cloud, this file must be named app.py.
- Submit to Canvas: a zip file containing app.py, requirements.txt, and
  Homework2_<StudentID>_link.txt (Streamlit Cloud URL and GitHub URL).
  Put those three files at the root of the zip, not inside a subfolder.
- Do NOT fine-tune a model. Do NOT scrape the web.
- Do NOT change DATASET_ID, MODEL_ID, N_TEST, or SPLIT_SEED.
- When you deploy, open Advanced settings and set Python to 3.12.
  If the app is already deployed, delete it and redeploy — the Python
  version cannot be changed in place.
- If the app calls a Hugging Face model, store a fine-grained Hugging Face
  API token in Streamlit Cloud secrets as HF_TOKEN. Do not put the token
  in app.py, requirements.txt, or any public file.
- First run may download the model and dataset (needs network). CPU is OK.

SECRET PHRASE (must appear on the app page): ISOM5160-HW2-SMS
"""

# STUDENT ANSWERS BLOCK
# Student ID: 21374430
# GenAI: OpenAI Codex helped interpret the requirements, write the code and answers,
# verify the app, and prepare the Git commands.
# Q1. A security analyst uses this app to review incoming SMS for suspected scams.
# Q2. The app classified 99 of 100 messages correctly (99% accuracy) and missed one spam text.
# I would use it to flag messages for review because this sample gives limited evidence for automatic blocking.
# A false positive could hide a bank login code, while a false negative could expose a customer to fraud.

from textwrap import shorten

import streamlit as st
from datasets import load_dataset
from transformers import pipeline

# Locked by the instructor — do not change these four values.
DATASET_ID = "ucirvine/sms_spam"
MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
N_TEST = 100
SPLIT_SEED = 42

# Course identity (required)
STUDENT_ID = "21374430"
COURSE_SECRET = "ISOM5160-HW2-SMS"  # do not change


def map_label(raw_label):
    """Map model or dataset labels to the two display names used in this app."""
    text = str(raw_label).strip().lower()
    if text in {"ham", "legitimate", "label_0", "0"}:
        return "legitimate"
    if text in {"spam", "smishing", "label_1", "1"}:
        return "spam"
    return str(raw_label)


@st.cache_resource
def load_model(token):
    """Load the SMS text-classification model once and reuse it."""
    # This function is provided. Keep device="cpu" and
    # model_kwargs={"low_cpu_mem_usage": False} so Streamlit Cloud can load
    # the model on CPU.
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=token,
        device="cpu",
        truncation=True,
        max_length=512,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    return classifier


@st.cache_data
def load_test_samples():
    """
    Load the Hugging Face dataset and take a fixed set of 100 samples.

    The Hub copy has only a train split, so this homework holds out 100
    rows after a shuffle with SPLIT_SEED. Do not change the seed or N_TEST,
    so every student is tested on the same 100 messages.
    """
    raw = load_dataset(DATASET_ID, split="train")
    picked = raw.shuffle(seed=SPLIT_SEED).select(range(N_TEST))
    texts = list(picked["sms"])
    true_labels = [map_label(label) for label in picked["label"]]
    return texts, true_labels


def classify_text(classifier, text):
    """Return (display_label, confidence) for one SMS string."""
    result = classifier(text)[0]
    return map_label(result["label"]), result["score"]


def compute_accuracy(pred_labels, true_labels):
    """Return accuracy as a float between 0 and 1."""
    matches = sum(pred == true for pred, true in zip(pred_labels, true_labels, strict=True))
    return matches / len(true_labels) if true_labels else 0.0


def main():
    st.title("ISOM5160")
    st.write("Homework 2 — SMS Spam / Smishing Alert")
    st.write(f"Student ID: {STUDENT_ID} | Course secret: {COURSE_SECRET}")

    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except FileNotFoundError:
        hf_token = None
    classifier = load_model(hf_token)

    alert_text = st.text_area(
        "SMS message",
        value=(
            "WINNER!! You have been selected to receive a $900 prize. "
            "Claim now at http://prize-alert.example or the offer expires."
        ),
        height=140,
    )

    if st.button("Classify SMS"):
        if alert_text.strip():
            label, confidence = classify_text(classifier, alert_text.strip())
            st.write(f"Prediction: {label} | Confidence: {confidence:.1%}")
        else:
            st.warning("Enter an SMS to classify.")

    st.subheader("Test on 100 Hugging Face samples")
    st.write(
        "Click the button to load 100 messages from the Hugging Face "
        "dataset and measure accuracy."
    )
    if st.button("Run 100-sample test"):
        texts, true_labels = load_test_samples()
        predictions, rows = [], []
        for i, (text, truth) in enumerate(zip(texts, true_labels, strict=True), start=1):
            label, confidence = classify_text(classifier, text)
            predictions.append(label)
            rows.append({
                "#": i,
                "Text (short)": shorten(text, width=90, placeholder="..."),
                "Ground truth": truth,
                "Prediction": label,
                "Confidence": f"{confidence:.1%}",
                "Match": label == truth,
            })
        accuracy = compute_accuracy(predictions, true_labels)
        st.metric("Accuracy", f"{accuracy:.0%}")
        st.dataframe(rows, hide_index=True)


if __name__ == "__main__":
    main()
