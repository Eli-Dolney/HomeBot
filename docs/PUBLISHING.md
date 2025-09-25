# Publishing Guide

Use this checklist to safely publish the repo:

1) Clean local data
```bash
bash scripts/clean_repo.sh
```

2) Prepare environment template (do not commit real values)
```bash
cp .env.example .env
# edit .env locally as needed; do NOT commit .env
```

3) Verify ignored files
```bash
git status
# Ensure logs/, sessions/, .venv/, *.wav, .env are not staged
```

4) Initialize Git and push
```bash
git init
git add .
git commit -m "Initial public release"
# create a new GitHub repo, then:
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

5) Optional: create a release
- Tag a version: `git tag v0.1.0 && git push --tags`
- Add release notes summarizing features and requirements

Security notes
- Keep `.env` private. Never include tokens/passwords in commits or issues.
- Logs may include transcripts; the `.gitignore` excludes them by default.
- Tools are local-only; review `tools.yaml` and `tools_m4.yaml` before enabling.
