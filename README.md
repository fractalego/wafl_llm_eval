# Code to test LLMs over the WAFL dataset.

Please run the code in the `src/` folder to test the LLMs over the [WAFL dataset](https://huggingface.co/datasets/fractalego/wafl-functions-dataset) dataset.
These are the results obtained up to now:

| LLM Name                               | Precision | Recall   | F1       |
|----------------------------------------|-----------|----------|----------|
| Phi-3-mini-4k-instruct (original)     | 1         | 0.92     | 0.96     |
| Mistral-7B-Instruct-v0.1 (original)   | 1         | 0.47     | 0.64     |
| Meta-Llama-3-8B-Instruct (original)   | 1         | 0.76     | 0.87     |
| Phi-3-mini-4k-instruct (after DPO)    | 1         | **0.95** | **0.97** |
| Mistral-7B-Instruct-v0.1 (after DPO)  | 0.93      | 0.73     | 0.82     |`
| Meta-Llama-3-8B-Instruct (after DPO)  | 0.91      | 0.87     | 0.89     |`