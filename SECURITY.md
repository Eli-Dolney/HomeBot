# Security & Privacy

This project is designed to run fully locally. Follow these guidelines before publishing:

- Do not commit secrets or personal data. Never commit `.env`.
- Keep `logs/` and `sessions/` out of Git (already in `.gitignore`).
- Validate tools: `tools.yaml` and `tools_m4.yaml` run local macOS commands only.
- Network: only `OLLAMA_HOST` on localhost is used. Avoid adding external API calls.
- Memory system is optional; if enabled, all data stays local. Clean periodically.

Reporting Issues:
- Open a GitHub issue without sharing sensitive logs; redact personal details first.
