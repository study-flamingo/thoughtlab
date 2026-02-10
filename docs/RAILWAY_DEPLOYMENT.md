# ThoughtLab Railway Deployment Guide

Deploy ThoughtLab to Railway with this step-by-step guide.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Railway Project                      │
├─────────────────┬─────────────────┬─────────────────────┤
│    Frontend     │     Backend     │       Redis         │
│   (nginx/React) │    (FastAPI)    │   (Railway Plugin)  │
│                 │                 │                     │
│ Dockerfile.     │   Dockerfile    │    Managed by       │
│ railway         │                 │    Railway          │
└────────┬────────┴────────┬────────┴──────────┬──────────┘
         │                 │                   │
         │                 └───────────────────┘
         │                         │
         │    ┌────────────────────┘
         │    │
         │    ▼
         │  ┌─────────────────────────────────────────────┐
         │  │              Neo4j Aura                      │
         │  │         (External - Free Tier)               │
         └──┤  https://console.neo4j.io                    │
            └─────────────────────────────────────────────┘
```

## Prerequisites

1. **Railway Account** — [railway.app](https://railway.app)
2. **Neo4j Aura Account** — [console.neo4j.io](https://console.neo4j.io) (free tier works)
3. **OpenAI API Key** (optional, for AI features)

---

## Step 1: Set Up Neo4j Aura (Free Tier)

1. Go to [console.neo4j.io](https://console.neo4j.io)
2. Create a new **Free** instance
3. Save your credentials:
   - **Connection URI**: `neo4j+s://xxxxxxxx.databases.neo4j.io`
   - **Username**: `neo4j`
   - **Password**: (generated)
4. Open Neo4j Browser and run the initialization script:
   ```bash
   # Copy contents of docker/neo4j/init.cypher and run in Neo4j Browser
   ```

> **Note**: Aura uses `neo4j+s://` (encrypted) instead of `bolt://`

---

## Step 2: Create Railway Project

1. Go to [railway.app](https://railway.app) and create a new project
2. Choose **"Empty Project"**

### Add Redis

1. Click **"+ New"** → **"Database"** → **"Redis"**
2. Railway provisions Redis automatically
3. Note: Redis URL will be available as `REDIS_URL` to linked services

### Add Backend Service

1. Click **"+ New"** → **"GitHub Repo"**
2. Select your ThoughtLab repository
3. Configure:
   - **Root Directory**: `backend`
   - **Build Command**: (auto-detected from Dockerfile)

4. Add environment variables:
   ```
   # Neo4j (from Aura)
   NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-aura-password
   
   # Redis (use Railway's reference)
   REDIS_URL=${{Redis.REDIS_URL}}
   
   # Security
   SECRET_KEY=generate-a-secure-random-string
   
   # Application
   ENVIRONMENT=production
   DEBUG=false
   
   # AI (optional)
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

5. Deploy and note the public URL (e.g., `https://thoughtlab-backend-xxxx.up.railway.app`)

### Add Frontend Service

1. Click **"+ New"** → **"GitHub Repo"**
2. Select the same ThoughtLab repository
3. Configure:
   - **Root Directory**: `frontend`
   - **Build Command**: (auto-detected from Dockerfile.railway)

4. Add **build-time** environment variable:
   ```
   # Point to your backend's public URL
   VITE_API_URL=https://thoughtlab-backend-xxxx.up.railway.app/api/v1
   ```

5. Deploy

---

## Step 3: Verify Deployment

1. **Backend Health Check**:
   ```bash
   curl https://thoughtlab-backend-xxxx.up.railway.app/health
   # Should return: {"status": "healthy", ...}
   ```

2. **Frontend**: Open the frontend URL in your browser

3. **API Docs**: Visit `https://thoughtlab-backend-xxxx.up.railway.app/docs`

---

## Environment Variables Reference

### Backend

| Variable | Required | Description |
|----------|----------|-------------|
| `NEO4J_URI` | ✅ | Neo4j connection URI (Aura: `neo4j+s://...`) |
| `NEO4J_USER` | ✅ | Neo4j username |
| `NEO4J_PASSWORD` | ✅ | Neo4j password |
| `REDIS_URL` | ✅ | Redis connection URL |
| `SECRET_KEY` | ✅ | Random secret for session security |
| `ENVIRONMENT` | ❌ | `production` or `development` |
| `DEBUG` | ❌ | `true` or `false` |
| `CORS_ORIGINS` | ❌ | Additional CORS origins (comma-separated) |
| `OPENAI_API_KEY` | ❌ | For AI features |
| `ANTHROPIC_API_KEY` | ❌ | For AI features |

### Frontend (Build-time)

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_API_URL` | ✅ | Backend API URL (include `/api/v1`) |

---

## Troubleshooting

### Backend won't connect to Neo4j Aura

- Ensure you're using `neo4j+s://` (not `bolt://`) for Aura
- Check that the instance is running in Aura console
- Verify credentials are correct

### Frontend can't reach backend (CORS errors)

- Railway domains (`*.up.railway.app`) are auto-allowed
- For custom domains, add them to `CORS_ORIGINS`

### Redis connection issues

- Use Railway's service reference: `${{Redis.REDIS_URL}}`
- Or copy the full URL from Railway's Redis service variables

### Build fails

- Check Railway build logs for specific errors
- Ensure `pyproject.toml` (backend) and `package.json` (frontend) are valid

---

## Custom Domains

1. In Railway, go to your service → **Settings** → **Domains**
2. Add your custom domain
3. Configure DNS as instructed by Railway
4. Update `CORS_ORIGINS` on backend if needed:
   ```
   CORS_ORIGINS=https://thoughtlab.yourdomain.com
   ```
5. Rebuild frontend with updated `VITE_API_URL` pointing to your custom backend domain

---

## Cost Considerations

- **Railway Free Tier**: 500 hours/month, $5 credit
- **Neo4j Aura Free**: 200k nodes, 400k relationships (plenty for development)
- **Redis**: Included in Railway usage

For production use, consider Railway's paid plans and Neo4j Aura Professional.

---

## Updating

Railway auto-deploys on push to your main branch. To manually redeploy:

1. Go to your service in Railway
2. Click **"Deploy"** → **"Trigger Deploy"**

Or use Railway CLI:
```bash
railway up
```
