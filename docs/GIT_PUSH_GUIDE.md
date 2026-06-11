# Git Push Guide — Two Contributors

This repo can be pushed so its history shows **two contributors**. What makes a
commit "belong" to a person is the **author name + email** on that commit — so
the two team members each commit their own part under their own identity.

> Use the email each person has on their **GitHub account** (or their GitHub
> `noreply` email) so GitHub links the commits to their profile and both show
> up on the repository's contributor graph.

Division of labor (matches the two zips in `dist/`):
- **Contributor 1 — backend / ML / data:** `backend/`, `ml/`, `scripts/`, `tests/`, requirements, `Dockerfile.backend`
- **Contributor 2 — frontend / docs / deploy:** `frontend/`, `docs/`, `notebooks/`, `README.md`, `docker-compose.yml`, `Dockerfile.frontend`

---

## Scenario A — one machine (simplest)

Both parts are already on this machine, so you can build the whole history here,
attributing each set of commits to the right person. Run from the project root.

```powershell
cd c:\Users\srikr\investment-intelligence
git init
git branch -M main

# --- Contributor 1: backend / ML ---
$A_NAME = "Contributor One"; $A_MAIL = "one@example.com"
git add ml/utils.py ml/synthetic.py backend/data_loader.py .gitignore .env.example requirements.txt requirements-deep.txt
git -c user.name=$A_NAME -c user.email=$A_MAIL commit -m "Data ingestion pipeline + project scaffolding"

git add ml/features.py ml/eda.py
git -c user.name=$A_NAME -c user.email=$A_MAIL commit -m "Feature engineering and EDA modules"

git add ml/models ml/training ml/evaluation
git -c user.name=$A_NAME -c user.email=$A_MAIL commit -m "Forecasting models, walk-forward training and evaluation"

git add ml/portfolio ml/risk ml/anomaly ml/explainability ml/recommendation
git -c user.name=$A_NAME -c user.email=$A_MAIL commit -m "Portfolio, risk, anomaly, explainability and recommendation engines"

git add ml/backtest.py ml/simulation.py ml/analytics.py ml/assistant.py ml/reports.py ml/__init__.py
git -c user.name=$A_NAME -c user.email=$A_MAIL commit -m "Bonus engines: backtest, simulation, analytics, assistant, reports"

git add backend
git -c user.name=$A_NAME -c user.email=$A_MAIL commit -m "FastAPI backend: schemas, services and routers"

git add tests pytest.ini scripts
git -c user.name=$A_NAME -c user.email=$A_MAIL commit -m "Test suite and offline pipeline scripts"

git add docker/Dockerfile.backend
git -c user.name=$A_NAME -c user.email=$A_MAIL commit -m "Backend Dockerfile"

# --- Contributor 2: frontend / docs / deploy ---
$B_NAME = "Contributor Two"; $B_MAIL = "two@example.com"
git add frontend/package.json frontend/next.config.mjs frontend/tsconfig.json frontend/tailwind.config.ts frontend/postcss.config.mjs frontend/.gitignore
git -c user.name=$B_NAME -c user.email=$B_MAIL commit -m "Frontend scaffolding: Next.js, Tailwind, TypeScript config"

git add frontend/lib frontend/components frontend/app/globals.css frontend/app/layout.tsx
git -c user.name=$B_NAME -c user.email=$B_MAIL commit -m "Shared UI primitives, charts and API client"

git add frontend/app/page.tsx frontend/app/stocks frontend/app/forecasting frontend/app/portfolio
git -c user.name=$B_NAME -c user.email=$B_MAIL commit -m "Dashboard, stock explorer, forecasting and portfolio pages"

git add frontend/app/risk frontend/app/anomalies frontend/app/insights frontend/app/settings
git -c user.name=$B_NAME -c user.email=$B_MAIL commit -m "Risk, anomaly, AI insights and settings pages"

git add docker/Dockerfile.frontend docker-compose.yml .dockerignore
git -c user.name=$B_NAME -c user.email=$B_MAIL commit -m "Dockerization and one-command compose deployment"

git add docs notebooks README.md PROJECT_STATUS.md
git -c user.name=$B_NAME -c user.email=$B_MAIL commit -m "Documentation, notebooks and project report"

# Anything not yet staged (catch-all)
git add -A
git -c user.name=$B_NAME -c user.email=$B_MAIL commit -m "Finalize repository" --allow-empty

# --- Push to GitHub ---
# Create an empty repo on github.com first (no README), then:
git remote add origin https://github.com/<your-org-or-user>/<repo>.git
git push -u origin main
```

---

## Scenario B — two people, two machines (uses the zips)

The most natural way to get two real contributors: each person works from their
own machine and GitHub account.

**Contributor 1** (has `dist/part1-backend-ml.zip`):
```bash
unzip part1-backend-ml.zip && cd investment-intelligence
git init && git branch -M main
git config user.name "Contributor One"
git config user.email "one@example.com"
git add -A
git commit -m "Backend, ML pipeline, tests and data ingestion"
# create the empty repo on GitHub, then:
git remote add origin https://github.com/<org>/<repo>.git
git push -u origin main
```

**Contributor 2** (has `dist/part2-frontend-docs.zip`):
```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
unzip ~/Downloads/part2-frontend-docs.zip -d .
# the zip has an investment-intelligence/ top folder — move its contents up if needed:
#   cp -r investment-intelligence/* . && rm -rf investment-intelligence
git config user.name "Contributor Two"
git config user.email "two@example.com"
git add -A
git commit -m "Next.js frontend, documentation and deployment"
git push
```

Result: the repo's history and contributor graph show both people, each having
pushed the part they owned.

---

## Notes

- **Don't commit generated/heavy folders.** `.gitignore` already excludes
  `.venv/`, `node_modules/`, `data/`, `ml/artifacts/`, `dist/`, build output.
- If you also want the **results** (the Kaggle artifacts, report, EDA charts)
  in the repo as deliverables, force-add them after the commits above:
  `git add -f ml/artifacts reports/generated && git commit -m "Add trained artifacts and report"`
- All commit messages here truthfully describe the files in that commit. Set the
  two identities to the real team members.
