# Evaluation Metrics in Tesstrain-GPU

This document explains the core metrics used to evaluate OCR model performance within the `tesstrain-gpu` framework. All calculations are handled in `tesstrain_core/metrics.py`.

## Core Mechanism: String Alignment
Before calculating any error rates, the system aligns the **Predicted String** with the **Ground Truth String** using the Levenshtein distance dynamic programming algorithm.

This algorithm maps out the minimum number of character edits required to transform the prediction into the ground truth. Every edit operation is classified as:
- **Match (M)**: The character was predicted correctly.
- **Substitution (S)**: The model predicted the wrong character.
- **Deletion (D)**: The model missed a character that exists in the ground truth.
- **Insertion (I)**: The model hallucinated a character that doesn't exist in the ground truth.

---

## 1. CER (Character Error Rate)
CER measures the character-level accuracy of the OCR model.

**Formula:**
$$CER = \frac{S + D + I}{N}$$

- **Numerator ($S + D + I$)**: Total number of substitutions, deletions, and insertions.
- **Denominator ($N$)**: Total number of characters in the Ground Truth.

> **Note on Edit Costs**: The framework uses a custom edit cost function for Lao text. Instead of all substitutions costing `1.0`, specific visually similar Lao character pairs (e.g., `ໃ` vs `ໄ`, `ດ` vs `ຕ`) are penalized less heavily with a substitution cost of `0.5`.

## 2. WER (Word Error Rate)
WER operates identically to CER but evaluates accuracy at the word level. Because written Lao does not naturally use spaces between words, the text is split by spaces (typically inserted at sentence boundaries or by a tokenizer) before alignment.

**Formula:**
$$WER = \frac{S_w + D_w + I_w}{N_w}$$

- Evaluates entire words as single tokens instead of individual characters.

## 3. CoER (Component-Level Error Rate)
CoER is a custom, highly granular metric designed to diagnose the **Vertical Stack Problem** in Lao script. Instead of a document-wide aggregate, it isolates Levenshtein edits by specific typographic categories (e.g., Tone Marks, Upper Vowels, Consonants).

**Formula for a given category $c$:**
$$CoER_c = \frac{S_c + D_c + I_c}{N_c}$$

- **Numerator ($S_c + D_c + I_c$)**: Count of every time a character belonging to category $c$ was substituted, deleted, or inserted.
  - *Substitutions & Deletions*: Based on the **Ground Truth** character's category.
  - *Insertions*: Based on the **Predicted** character's category (since inserted characters do not exist in the ground truth).
- **Denominator ($N_c$)**: The total number of characters belonging to category $c$ that actually exist in the Ground Truth sentence.

> **Important**: Because insertions increase the error numerator but do not increase the ground truth denominator ($N_c$), it is mathematically possible for a category's CoER to exceed 1.0 (100%) if the model frequently hallucinates characters in that category.

## 4. ACC (Sequence / Line Accuracy)
ACC is a strict, binary metric that measures the percentage of entire image sequences (sentences or text lines) that were predicted perfectly.

**Formula:**
$$ACC = \frac{\text{Perfectly Predicted Sequences}}{\text{Total Sequences}}$$

- Even a single punctuation or spacing mistake invalidates the entire sequence, making this the most rigorous benchmark of complete model accuracy.
