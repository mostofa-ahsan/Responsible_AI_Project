## Method: QLoRA

All fine-tuned models are QLoRA adapters: the base model is frozen and quantized to 4-bit NF4 (double quantization, bf16 compute) during training, and only a bf16 LoRA adapter of rank 16 (alpha 32, dropout 0.05) on the q, k, v, o, gate, up and down projections is trained: 0.44–0.53% of the parameters (Qwen3-8B 43.6M of 8.19B, Gemma 4 E4B 34.9M of 8.00B, Llama 3.1 8B 41.9M of 8.03B), stored in 70–87 MB per adapter (bf16 safetensors; the run 10-03 adapters hold the same parameters in fp32). At inference the adapter is applied to the bf16 base (vLLM) unless stated otherwise.

## Trained-question recall

The exact-question test asks 500 training questions verbatim (stratified by question type × dimension; all 279 sources of the paraphrased seen-facts test are included, so those items form exact/paraphrase pairs). It measures recall of trained content and must be read together with the paraphrase test (robustness to rewording) and the unseen-document test (generalization): Table 7, Table 7b, Table 7c and Figure 9.
- Qwen3-8B, exact questions, judge accuracy: base – (KF recall –, exact repro. 0.000); concise base (<= 40 words) – (KF recall –, exact repro. 0.004); ep1 – (KF recall –, exact repro. 0.054); ep2 – (KF recall –, exact repro. 0.184); ep3 – (KF recall –, exact repro. 0.214).
- Gemma 4 E4B, exact questions, judge accuracy: .
- Llama 3.1 8B, exact questions, judge accuracy: .
- Robustness gap (judge accuracy on the exact question minus its paraphrase, 279 pairs): Qwen3-8B base +nan [+nan, +nan]; Qwen3-8B ep1 +nan [+nan, +nan]; Qwen3-8B ep2 +nan [+nan, +nan]; Qwen3-8B ep3 +nan [+nan, +nan] (`results/trained_eval/robustness_gap.csv`).
- Deployment (Table 8, Figure 10): .
