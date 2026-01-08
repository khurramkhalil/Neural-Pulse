### **Project Design Specification: Runtime STL Defense Against SECA**

#### **1. Problem-Gap-Solution (The Abstract)**

**The Problem:** Large Language Models (LLMs) have been shown to be brittle against **Semantically Equivalent and Coherent Attacks (SECA)**. Unlike traditional "jailbreaks" that use gibberish or obvious role-play, SECA attacks use verbose, lexically diverse, and natural-sounding rephrasings of valid questions to force the model into a hallucinated state. Because these attacks satisfy semantic equivalence constraints, they easily bypass static "input filters" (like the Feasibility Checker) and "output filters" (which only catch the error after the damage is done).
**The Gap:** Current defenses suffer from two critical limitations: they are either **static** (analyzing the prompt before execution) or **circular** (relying on the model’s own high confidence, as seen in recent calibration works like Mao et al., which fails when the model is "confidently wrong"). There is currently no framework that treats the *process* of hallucination as a detectable temporal anomaly. We lack a mechanism to monitor the **"Adversarial Stress"**—the internal cognitive dissonance and semantic drift—that occurs *during* the generation of a forced hallucination.
**The Solution:** We propose a **Runtime Verification Framework using Signal Temporal Logic (STL)**. Instead of checking the text, we monitor the continuous "vital signs" of the generation process—specifically **Information Entropy** (internal uncertainty) and **Semantic Entailment** (external grounding)—as time-series signals. We hypothesize that SECA attacks induce a distinct temporal signature: a "waffling" phase characterized by high verbosity and entropy spikes, followed by a collapse in semantic entailment. By encoding these signatures into robust STL formulas (e.g., *“The entropy must not remain high while entailment drops”*), we can detect and intercept attacks in real-time. This offers a **novel, non-circular defense** that verifies the *validity of the reasoning process*, not just the input or output.

---

#### **2. Research Implementation Plan**

Here is the structured roadmap for the coding and research phase. The goal is to move from **Signal Extraction** to **Formula Mining** and finally **Evaluation**.

### **Phase 1: The "Oscilloscope" (Data & Signal Extraction)**

**Goal:** Generate the datasets and extract the raw time-series traces. We need to "see" what an attack looks like mathematically compared to a normal response.

* **Step 1.1: Attack Reproduction**
* **Input:** Use the open-sourced SECA dataset (adversarial prompts) and the original MMLU dataset (clean prompts).
* **Target Model:** Set up Llama-3-8B (or similar) as the victim.
* **Action:** Run inference on both datasets.


* **Step 1.2: Instrumentation (The Monitor)**
* As the model generates token , you must log three signals at every step:


1. **Entropy Signal ():** Calculate the Shannon entropy of the softmax distribution. *(Hypothesis: SECA causes spikes here).*
2. **Attention Signal ():** Extract attention weights from the final layer. Log the ratio of attention paid to the *Context* vs. *Generated Tokens*. *(Hypothesis: SECA causes "Context Detachment").*
3. **Entailment Signal ():** Since cost is no object, run a DeBERTa-v3-large model in parallel. Score the entailment of the current sentence against the original system prompt. *(Hypothesis: SECA causes a sharp drop in entailment).*



### **Phase 2: Diagnosis & Formula Mining**

**Goal:** Visually and mathematically identify the difference between `Trace_Normal` and `Trace_Attack`.

* **Step 2.1: Visualization**
* Plot the traces (-axis = Time/Tokens, -axis = Signal Value).
* Overlay the "Clean" traces (Blue) and "Attack" traces (Red).
* **Deliverable:** A plot showing the "SECA Signature"—likely a long period of high entropy (waffling) followed by a sudden divergence in entailment.


* **Step 2.2: STL Specification**
* Use a library like `MoonLight` or `RTAMT`.
* Define your "Anti-Waffle" formulas.
* *Draft Formula:* 
* *Tuning:* Use the plots to find the correct thresholds (e.g., Is "High Entropy" > 0.5 or > 0.8?).



### **Phase 3: Evaluation (The Defense)**

**Goal:** Prove that the STL monitor catches attacks that static checkers miss.

* **Step 3.1: The Battle**
* Run the full SECA dataset through your STL Monitor.
* **Metric:** Calculate True Positive Rate (Detection of Attack) vs. False Positive Rate (Flagging a normal hard question).


* **Step 3.2: Comparison**
* Compare your runtime detector against a baseline "Perplexity Filter" (a common, weaker defense).
* Show that your method works because it accounts for *time* (the trajectory), whereas perplexity just looks at the average.



### **Phase 4: Optimization (Optional/Future Work)**

* If the NLI model (Entailment signal) is too slow, see if you can achieve the same accuracy using *only* Entropy and Attention (the "Zero-Overhead" approach).

**Where to Start Coding:**
Start with **Phase 1, Step 1.2**. Write a Python script that hooks into the Hugging Face `generate()` loop and prints out the **Entropy** of the next token. If you can see the entropy spiking when you trick the model, you have a thesis.
