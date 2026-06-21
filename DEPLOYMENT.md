# Deployment Guide — Streamlit Community Cloud (free)

This gets you a **public URL** (e.g. `https://your-app-name.streamlit.app`) you can
hand to a recruiter. No credit card, no server to manage.

---

## 1. Push the project to GitHub

```bash
cd rag-qa-bot
git init
git add .
git commit -m "RAG document Q&A bot: hybrid retrieval + grounding guardrail"
git branch -M main
git remote add origin https://github.com/Shivateja832/rag-qa-bot.git
git push -u origin main
```

> Double check `.gitignore` is excluding `.env`, `.streamlit/secrets.toml`, and
> `vector_store/` before you push — confirm with `git status` that none of
> these show up as files about to be committed.

## 2. Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **"New app"**.
3. Pick your `rag-qa-bot` repo, branch `main`, main file path `app.py`.
4. Click **"Advanced settings"** before deploying:
   - **Secrets**: paste the contents of `.streamlit/secrets.toml.example`,
     but replace the placeholder with your real Gemini key:
     ```toml
     GEMINI_API_KEY = "AIza...your real key..."
     ```
   - Python version: 3.11 or 3.12.
5. Click **Deploy**.

## 3. First load builds the index automatically

The first time the deployed app loads, `app.py` detects there's no index yet
(the `vector_store/` directory isn't committed to git on purpose — it's a
generated artifact) and builds it automatically: ingests the 5 documents in
`data/`, chunks them, downloads `all-MiniLM-L6-v2` (~80MB, one-time), embeds,
and writes the ChromaDB + BM25 index. This takes **about a minute** and only
happens once per deployment (subsequent app restarts on Streamlit Cloud reuse
the same container's disk during the session, but a full redeploy will
rebuild it again — that's expected and fine).

## 4. Get your free Gemini API key (if you don't have one yet)

1. Go to **aistudio.google.com/apikey**.
2. Sign in with a Google account.
3. Click **"Create API key"** — no credit card required.
4. Copy the key into the Streamlit Cloud secrets box from step 2.

## 5. Verify the deployment

Once deployed, open the app URL and ask:
- *"How much have solar panel costs fallen since 2010, and why?"* — should
  answer with a citation to `Renewable_Energy_Transition_Report.pdf`.
- *"What's the capital of France?"* — should explicitly decline, since this
  is outside the document collection (proves the grounding guardrail works
  in production, not just locally).

If both work, you have a fully deployed, publicly accessible RAG application.

## 6. Before sending the link to a recruiter

Run the retrieval evaluation locally once and paste the output into your README or screen recording — this is the single most distinguishing thing you can show on top of "the demo works":

```bash
python scripts/eval_retrieval.py --ablation
```

A recruiter reading 50+ submissions will skim past another working chat demo. A Hit@K / MRR table, with a dense-vs-BM25-vs-hybrid ablation backing up *why* hybrid retrieval was chosen, is the kind of evidence that signals you understand retrieval as a measurable system, not just a working pipeline.

---

## Alternative: Hugging Face Spaces (also free)

If you'd rather deploy there instead: create a new Space, choose
**Streamlit** as the SDK, push this same repo to the Space's git remote, and
add `GEMINI_API_KEY` under the Space's **Settings → Repository secrets**. The
`_get_setting()` helper in `src/config.py` only checks `st.secrets` and
`os.getenv`, both of which HF Spaces populates the same way Streamlit Cloud
does, so no code changes are needed.
