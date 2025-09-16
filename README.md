# AI Agency OS – n8n Backend

This repo deploys **n8n** (workflow automation) to Render using Docker + Postgres.

## 🚀 Deployment Steps
1. Fork this repo to your GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com/).
3. Click **New + → Blueprint**.
4. Connect GitHub → select this repo.
5. Deploy with `render.yaml`.

## 🔑 Required Secrets
- `N8N_BASIC_AUTH_USER` (set in Render)
- `N8N_BASIC_AUTH_PASSWORD` (set in Render)
- `N8N_ENCRYPTION_KEY` (recommended for secrets)

## 📍 URL
App will be deployed at:
`https://ai-agency-os-n8n.onrender.com`
