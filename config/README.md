Place your local credentials in the files below. These files are ignored by Git.

Files:
- `config/kaggle_credentials.json`
- `config/qwen_config.json`

Create them by copying the corresponding `*.example.json` files in this folder and then filling in the values.

Kaggle:
- `username`: your Kaggle username
- `key`: your Kaggle API key

Qwen:
- `base_url`: OpenAI-compatible endpoint base URL, for example `http://localhost:11434/v1`
- `api_key`: API key if your endpoint requires one; otherwise any non-empty placeholder is fine
- `model`: your Qwen 2.5 model id, for example `qwen2.5:7b`
