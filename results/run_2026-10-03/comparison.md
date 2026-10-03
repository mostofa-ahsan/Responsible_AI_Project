# Closed-book comparison: base vs QLoRA fine-tuned (run_2026-10-03)

Judge accuracy: Opus 5.5 grading against the reference answer (correct 1, partial 0.5) on a fixed stratified subset (same questions for every model and arm). Token F1 and ROUGE-L use all test questions. Brackets are bootstrap 95% CIs; Δ is fine-tuned minus base, paired bootstrap over the same questions.

## test_indomain (unseen chunks of training documents)

| Model | Arm | Judge accuracy | Token F1 | ROUGE-L | n (graded / all) |
|---|---|---|---|---|---|
| qwen3-8b | base | 0.559 [0.527, 0.591] | 0.265 [0.259, 0.272] | 0.183 [0.177, 0.189] | 500 / 791 |
| qwen3-8b | finetuned | 0.445 [0.415, 0.473] | 0.375 [0.366, 0.383] | 0.285 [0.276, 0.293] | 500 / 791 |
| qwen3-8b | **Δ** | **-0.114 [-0.144, -0.085]** | 0.110 [0.101, 0.118] | 0.102 [0.095, 0.109] | |
| gemma-4-e4b-it | base | 0.588 [0.555, 0.621] | 0.177 [0.172, 0.183] | 0.119 [0.115, 0.123] | 500 / 791 |
| gemma-4-e4b-it | finetuned | 0.438 [0.409, 0.466] | 0.382 [0.373, 0.390] | 0.294 [0.285, 0.303] | 500 / 791 |
| gemma-4-e4b-it | **Δ** | **-0.150 [-0.180, -0.119]** | 0.204 [0.195, 0.214] | 0.175 [0.166, 0.184] | |
| llama-3.1-8b-instruct | base | 0.520 [0.488, 0.552] | 0.182 [0.176, 0.189] | 0.130 [0.125, 0.135] | 500 / 791 |
| llama-3.1-8b-instruct | finetuned | 0.435 [0.406, 0.465] | 0.375 [0.367, 0.383] | 0.286 [0.278, 0.295] | 500 / 791 |
| llama-3.1-8b-instruct | **Δ** | **-0.085 [-0.114, -0.055]** | 0.193 [0.183, 0.202] | 0.156 [0.148, 0.165] | |

## test_heldout_docs (unseen documents)

| Model | Arm | Judge accuracy | Token F1 | ROUGE-L | n (graded / all) |
|---|---|---|---|---|---|
| qwen3-8b | base | 0.506 [0.474, 0.540] | 0.258 [0.253, 0.262] | 0.175 [0.172, 0.179] | 500 / 1486 |
| qwen3-8b | finetuned | 0.370 [0.342, 0.400] | 0.353 [0.347, 0.360] | 0.266 [0.260, 0.272] | 500 / 1486 |
| qwen3-8b | **Δ** | **-0.136 [-0.164, -0.109]** | 0.096 [0.090, 0.102] | 0.091 [0.086, 0.096] | |
| gemma-4-e4b-it | base | 0.538 [0.504, 0.573] | 0.179 [0.175, 0.184] | 0.120 [0.117, 0.123] | 500 / 1486 |
| gemma-4-e4b-it | finetuned | 0.389 [0.359, 0.420] | 0.359 [0.353, 0.366] | 0.274 [0.267, 0.280] | 500 / 1486 |
| gemma-4-e4b-it | **Δ** | **-0.149 [-0.178, -0.120]** | 0.180 [0.173, 0.186] | 0.154 [0.148, 0.160] | |
| llama-3.1-8b-instruct | base | 0.457 [0.423, 0.489] | 0.180 [0.176, 0.185] | 0.128 [0.125, 0.131] | 500 / 1486 |
| llama-3.1-8b-instruct | finetuned | 0.359 [0.330, 0.388] | 0.345 [0.339, 0.351] | 0.258 [0.252, 0.264] | 500 / 1486 |
| llama-3.1-8b-instruct | **Δ** | **-0.098 [-0.126, -0.068]** | 0.164 [0.158, 0.171] | 0.130 [0.124, 0.135] | |

## test_seen_facts (new paraphrases of TRAINING questions: facts the fine-tuned model saw)

| Model | Arm | Judge accuracy | Token F1 | ROUGE-L | n (graded / all) |
|---|---|---|---|---|---|
| qwen3-8b | base | 0.504 [0.459, 0.547] | 0.213 [0.203, 0.222] | 0.141 [0.134, 0.149] | 279 / 279 |
| qwen3-8b | finetuned | 0.426 [0.387, 0.466] | 0.326 [0.311, 0.341] | 0.240 [0.226, 0.255] | 279 / 279 |
| qwen3-8b | **Δ** | **-0.077 [-0.118, -0.036]** | 0.113 [0.098, 0.127] | 0.100 [0.087, 0.112] | |
| gemma-4-e4b-it | base | 0.512 [0.468, 0.556] | 0.150 [0.143, 0.158] | 0.101 [0.095, 0.107] | 279 / 279 |
| gemma-4-e4b-it | finetuned | 0.409 [0.373, 0.444] | 0.316 [0.303, 0.331] | 0.236 [0.223, 0.250] | 279 / 279 |
| gemma-4-e4b-it | **Δ** | **-0.104 [-0.149, -0.061]** | 0.166 [0.152, 0.180] | 0.135 [0.123, 0.148] | |
| llama-3.1-8b-instruct | base | 0.450 [0.409, 0.491] | 0.154 [0.145, 0.163] | 0.107 [0.101, 0.114] | 279 / 279 |
| llama-3.1-8b-instruct | finetuned | 0.432 [0.396, 0.471] | 0.342 [0.325, 0.358] | 0.260 [0.243, 0.277] | 279 / 279 |
| llama-3.1-8b-instruct | **Δ** | **-0.018 [-0.059, 0.023]** | 0.187 [0.172, 0.202] | 0.152 [0.138, 0.167] | |

Unsupported claims per answer (judge count vs the reference; this split only):

| Model | Arm | 0 | 1 | 2+ |
|---|---|---|---|---|
| qwen3-8b | base | 4 (1%) | 24 (9%) | 251 (90%) |
| qwen3-8b | finetuned | 45 (16%) | 113 (41%) | 121 (43%) |
| gemma-4-e4b-it | base | 23 (8%) | 9 (3%) | 247 (89%) |
| gemma-4-e4b-it | finetuned | 74 (27%) | 95 (34%) | 110 (39%) |
| llama-3.1-8b-instruct | base | 13 (5%) | 13 (5%) | 253 (91%) |
| llama-3.1-8b-instruct | finetuned | 42 (15%) | 116 (42%) | 121 (43%) |

## Improvement in judge accuracy by question type and dimension (Δ, paired 95% CI)

**gemma-4-e4b-it, test_heldout_docs**

| Group | Δ judge accuracy | n |
|---|---|---|
| application | -0.149 [-0.207, -0.085] | 94 |
| definition | -0.123 [-0.172, -0.074] | 142 |
| explanation | -0.250 [-0.309, -0.195] | 136 |
| factual | -0.070 [-0.129, -0.016] | 128 |
| assessment | -0.155 [-0.276, -0.035] | 29 |
| equity_accessibility | -0.177 [-0.250, -0.097] | 62 |
| faculty_readiness | 0.167 [0.000, 0.500] | 6 |
| general | -0.198 [-0.278, -0.119] | 63 |
| governance | -0.190 [-0.260, -0.114] | 79 |
| leadership_strategy | -0.077 [-0.192, 0.019] | 26 |
| monitoring_improvement | -0.226 [-0.306, -0.145] | 31 |
| privacy_security | -0.097 [-0.208, 0.014] | 36 |
| procurement_technology | -0.115 [-0.269, 0.019] | 26 |
| student_ai_literacy | -0.093 [-0.222, 0.037] | 27 |
| teaching_learning | -0.126 [-0.191, -0.061] | 115 |

**gemma-4-e4b-it, test_indomain**

| Group | Δ judge accuracy | n |
|---|---|---|
| application | -0.109 [-0.168, -0.050] | 101 |
| definition | -0.158 [-0.218, -0.103] | 117 |
| explanation | -0.253 [-0.303, -0.203] | 150 |
| factual | -0.057 [-0.125, 0.011] | 132 |
| assessment | -0.150 [-0.250, -0.050] | 30 |
| equity_accessibility | -0.162 [-0.311, -0.013] | 37 |
| faculty_readiness | -0.111 [-0.211, 0.000] | 45 |
| general | -0.167 [-0.242, -0.100] | 60 |
| governance | -0.167 [-0.235, -0.099] | 81 |
| leadership_strategy | -0.125 [-0.219, -0.031] | 32 |
| monitoring_improvement | -0.115 [-0.250, 0.038] | 26 |
| privacy_security | -0.217 [-0.317, -0.117] | 30 |
| procurement_technology | -0.263 [-0.474, -0.079] | 19 |
| student_ai_literacy | -0.083 [-0.188, 0.021] | 48 |
| teaching_learning | -0.147 [-0.217, -0.071] | 92 |

**gemma-4-e4b-it, test_seen_facts**

| Group | Δ judge accuracy | n |
|---|---|---|
| application | -0.111 [-0.191, -0.032] | 63 |
| definition | -0.141 [-0.225, -0.056] | 71 |
| explanation | -0.113 [-0.200, -0.027] | 75 |
| factual | -0.050 [-0.143, 0.043] | 70 |
| assessment | -0.167 [-0.375, 0.042] | 12 |
| equity_accessibility | -0.140 [-0.260, -0.020] | 25 |
| faculty_readiness | -0.065 [-0.196, 0.065] | 23 |
| general | -0.067 [-0.200, 0.083] | 30 |
| governance | -0.167 [-0.260, -0.073] | 48 |
| leadership_strategy | -0.233 [-0.400, -0.100] | 15 |
| monitoring_improvement | -0.048 [-0.214, 0.095] | 21 |
| privacy_security | -0.184 [-0.316, -0.053] | 19 |
| procurement_technology | -0.150 [-0.350, 0.050] | 10 |
| student_ai_literacy | 0.000 [-0.160, 0.180] | 25 |
| teaching_learning | -0.049 [-0.167, 0.078] | 51 |

**llama-3.1-8b-instruct, test_heldout_docs**

| Group | Δ judge accuracy | n |
|---|---|---|
| application | -0.096 [-0.154, -0.037] | 94 |
| definition | -0.053 [-0.109, 0.000] | 142 |
| explanation | -0.188 [-0.239, -0.132] | 136 |
| factual | -0.055 [-0.113, 0.004] | 128 |
| assessment | 0.000 [-0.086, 0.086] | 29 |
| equity_accessibility | -0.153 [-0.234, -0.073] | 62 |
| faculty_readiness | 0.167 [0.000, 0.500] | 6 |
| general | -0.143 [-0.222, -0.064] | 63 |
| governance | -0.120 [-0.203, -0.032] | 79 |
| leadership_strategy | 0.019 [-0.115, 0.154] | 26 |
| monitoring_improvement | -0.129 [-0.226, -0.048] | 31 |
| privacy_security | -0.125 [-0.222, -0.028] | 36 |
| procurement_technology | -0.077 [-0.192, 0.038] | 26 |
| student_ai_literacy | -0.056 [-0.167, 0.056] | 27 |
| teaching_learning | -0.091 [-0.152, -0.035] | 115 |

**llama-3.1-8b-instruct, test_indomain**

| Group | Δ judge accuracy | n |
|---|---|---|
| application | -0.069 [-0.129, -0.010] | 101 |
| definition | -0.107 [-0.162, -0.047] | 117 |
| explanation | -0.130 [-0.180, -0.080] | 150 |
| factual | -0.026 [-0.099, 0.042] | 132 |
| assessment | -0.100 [-0.200, 0.000] | 30 |
| equity_accessibility | -0.189 [-0.297, -0.095] | 37 |
| faculty_readiness | -0.044 [-0.167, 0.078] | 45 |
| general | -0.067 [-0.150, 0.017] | 60 |
| governance | -0.080 [-0.148, -0.018] | 81 |
| leadership_strategy | -0.109 [-0.219, 0.000] | 32 |
| monitoring_improvement | -0.115 [-0.250, 0.019] | 26 |
| privacy_security | -0.150 [-0.267, -0.033] | 30 |
| procurement_technology | -0.105 [-0.237, 0.026] | 19 |
| student_ai_literacy | 0.031 [-0.073, 0.135] | 48 |
| teaching_learning | -0.092 [-0.163, -0.016] | 92 |

**llama-3.1-8b-instruct, test_seen_facts**

| Group | Δ judge accuracy | n |
|---|---|---|
| application | -0.087 [-0.151, -0.024] | 63 |
| definition | 0.035 [-0.049, 0.120] | 71 |
| explanation | -0.053 [-0.140, 0.027] | 75 |
| factual | 0.029 [-0.057, 0.114] | 70 |
| assessment | 0.042 [-0.125, 0.208] | 12 |
| equity_accessibility | -0.120 [-0.240, 0.000] | 25 |
| faculty_readiness | 0.043 [-0.130, 0.239] | 23 |
| general | 0.017 [-0.100, 0.133] | 30 |
| governance | -0.073 [-0.156, 0.010] | 48 |
| leadership_strategy | -0.067 [-0.200, 0.067] | 15 |
| monitoring_improvement | 0.000 [-0.143, 0.167] | 21 |
| privacy_security | -0.079 [-0.237, 0.079] | 19 |
| procurement_technology | -0.150 [-0.300, -0.050] | 10 |
| student_ai_literacy | 0.120 [-0.020, 0.260] | 25 |
| teaching_learning | 0.010 [-0.088, 0.118] | 51 |

**qwen3-8b, test_heldout_docs**

| Group | Δ judge accuracy | n |
|---|---|---|
| application | -0.080 [-0.149, -0.016] | 94 |
| definition | -0.127 [-0.176, -0.077] | 142 |
| explanation | -0.199 [-0.254, -0.147] | 136 |
| factual | -0.121 [-0.180, -0.062] | 128 |
| assessment | -0.052 [-0.190, 0.069] | 29 |
| equity_accessibility | -0.185 [-0.258, -0.113] | 62 |
| faculty_readiness | 0.250 [0.000, 0.583] | 6 |
| general | -0.175 [-0.246, -0.095] | 63 |
| governance | -0.177 [-0.253, -0.101] | 79 |
| leadership_strategy | 0.019 [-0.077, 0.115] | 26 |
| monitoring_improvement | -0.113 [-0.210, -0.016] | 31 |
| privacy_security | -0.167 [-0.264, -0.069] | 36 |
| procurement_technology | -0.115 [-0.250, 0.019] | 26 |
| student_ai_literacy | -0.111 [-0.222, 0.000] | 27 |
| teaching_learning | -0.143 [-0.204, -0.083] | 115 |

**qwen3-8b, test_indomain**

| Group | Δ judge accuracy | n |
|---|---|---|
| application | -0.114 [-0.168, -0.054] | 101 |
| definition | -0.107 [-0.162, -0.051] | 117 |
| explanation | -0.163 [-0.213, -0.113] | 150 |
| factual | -0.064 [-0.129, 0.000] | 132 |
| assessment | -0.100 [-0.200, 0.000] | 30 |
| equity_accessibility | -0.108 [-0.216, 0.000] | 37 |
| faculty_readiness | -0.011 [-0.089, 0.067] | 45 |
| general | -0.167 [-0.250, -0.092] | 60 |
| governance | -0.148 [-0.216, -0.080] | 81 |
| leadership_strategy | -0.156 [-0.266, -0.062] | 32 |
| monitoring_improvement | -0.231 [-0.346, -0.115] | 26 |
| privacy_security | -0.117 [-0.233, -0.017] | 30 |
| procurement_technology | -0.079 [-0.237, 0.053] | 19 |
| student_ai_literacy | -0.104 [-0.208, 0.000] | 48 |
| teaching_learning | -0.071 [-0.147, 0.005] | 92 |

**qwen3-8b, test_seen_facts**

| Group | Δ judge accuracy | n |
|---|---|---|
| application | -0.064 [-0.127, 0.008] | 63 |
| definition | -0.113 [-0.190, -0.035] | 71 |
| explanation | -0.087 [-0.180, 0.000] | 75 |
| factual | -0.043 [-0.114, 0.036] | 70 |
| assessment | 0.000 [-0.208, 0.208] | 12 |
| equity_accessibility | -0.120 [-0.240, 0.000] | 25 |
| faculty_readiness | 0.000 [-0.174, 0.196] | 23 |
| general | -0.033 [-0.167, 0.100] | 30 |
| governance | -0.115 [-0.208, -0.021] | 48 |
| leadership_strategy | -0.067 [-0.233, 0.067] | 15 |
| monitoring_improvement | -0.095 [-0.262, 0.071] | 21 |
| privacy_security | -0.105 [-0.237, 0.000] | 19 |
| procurement_technology | -0.200 [-0.450, 0.050] | 10 |
| student_ai_literacy | -0.020 [-0.120, 0.080] | 25 |
| teaching_learning | -0.088 [-0.176, 0.000] | 51 |

## Training

| Model | Epochs | Steps | Train time (min) | Peak VRAM (GB) | Tokens/s | Best val loss | Adapter (MB) |
|---|---|---|---|---|---|---|---|
| qwen3-8b | 1 | 1249 | 67.8 | 11.4 | 546 | 1.87867 | 186.1 |
| gemma-4-e4b-it | 1 | 1249 | 65.8 | 19.8 | 531 | 1.96345 | 171.8 |
| llama-3.1-8b-instruct | 1 | 1249 | 69.1 | 10.3 | 615 | 2.01983 | 185.1 |

Shared recipe: QLoRA (4-bit NF4, bf16 compute), LoRA r 16 / alpha 32 / dropout 0.05 on all language-model linear layers, lr 2e-4 cosine, effective batch 16, max length 1,024, seed 42, closed-book format (answer only), paraphrases as extra examples, each model's own chat template; best epoch by validation loss.

**Post-training quantization:** merged models were not quantized in this run (disk). AWQ and GGUF exports will be produced for the best model only.
