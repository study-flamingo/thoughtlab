# Agent Session Notes

This file tracks important changes, debugging sessions, and configuration updates made during Claude Code agent sessions.

---

## 2025-12-28: CORS Configuration, Redis Optional, and Windows Script Improvements

### Issues Resolved

1. **CORS Errors Blocking Frontend-Backend Communication**
   - Frontend couldn't connect to backend API
   - Error: `Access-Control-Allow-Origin header not present`
   - Root cause: Backend was not running or using wrong configuration

2. **Redis Dependency Causing Backend Crashes**
   - Backend required Redis to start but Redis wasn't running
   - Made Redis completely optional for development
   - Backend now starts successfully with just Neo4j

3. **Port Management Issues**
   - Vite was auto-incrementing to ports 5174-5179 when 5173 was occupied
   - Old backend processes were occupying port 8000
   - Stop scripts weren't reliably killing all processes on Windows

4. **Windows Process Management Problems**
   - Git Bash scripts couldn't reliably kill Windows processes
   - `netstat` parsing was inconsistent
   - Processes remained orphaned after script execution

### Changes Made

#### Backend Configuration

**File: `backend/app/core/config.py`**
- Added ports 5173-5179 to CORS allowed origins list
- Made `redis_url` optional with default value
- Allows frontend to connect from any Vite dev server port

**File: `backend/app/main.py`**
- Made Redis connection optional in startup lifecycle (lines 34-38)
- Backend continues if Redis unavailable: "Continuing without cache"
- Updated health check to show Redis as "not configured (optional)"
- Overall health status no longer depends on Redis

#### Frontend Configuration

**File: `frontend/vite.config.ts`**
- Added `strictPort: true` to Vite config
- Prevents auto-incrementing to 5174, 5175, etc.
- Forces port 5173 or fails with clear error message

#### Script Improvements

**Bash Scripts Enhanced (Git Bash):**

**File: `start.sh`, `stop.sh`, `restart.sh`**
- Added PATH setup for Git Bash utilities (`/usr/bin`, `/mingw64/bin`)
- Added command availability checks with helpful error messages
- Enhanced `stop.sh` with multiple kill methods:
  - Port-based killing using `netstat` and `taskkill`
  - Process name killing using PowerShell queries
  - Process tree termination with `taskkill //T`
  - Kills ports 5173-5179 to catch stray Vite processes

**PowerShell Scripts Created (Native Windows):**

**New Files: `start.ps1`, `stop.ps1`, `restart.ps1`**
- Created native PowerShell versions for better Windows compatibility
- Uses `Get-NetTCPConnection` for reliable port detection
- Uses `Get-WmiObject` to find processes by command line
- More reliable process killing with `Stop-Process -Force`
- Better error handling and output formatting
- Recommended for Windows users over Bash scripts

### Architecture Notes

#### CORS Configuration Strategy
The backend now allows requests from:
- Explicit origins: `http://localhost:5173` through `http://localhost:5179`
- Regex pattern: Any localhost or 127.0.0.1 with any port
- This dual approach ensures compatibility during development

#### Redis Optional Design Pattern
```
Startup Flow:
1. Connect to Neo4j (required) → Retry 3 times, fail if unsuccessful
2. Connect to Redis (optional) → Try once, continue if fails
3. Application startup complete

Health Check:
- Neo4j: healthy/unhealthy → Affects overall status
- Redis: healthy/not configured/not available → Does NOT affect overall status
```

#### Process Management Strategy
Windows processes are killed using three methods (in order):
1. **Port-based killing** - Find PID listening on port, kill with taskkill
2. **Process name search** - Find all python.exe/node.exe with specific command line
3. **PID file cleanup** - Kill using stored PID from `.run/` directory

This redundancy ensures processes are reliably terminated even if one method fails.

### Development Workflow

**Recommended Approach (Windows):**

Use PowerShell scripts for best reliability:
```powershell
# Stop all servers
.\stop.ps1

# Start both servers (background)
.\start.ps1

# Restart servers
.\restart.ps1
```

**Alternative Approach (Manual Control):**

Run servers in separate terminals for easier debugging:

Terminal 1 - Backend:
```bash
cd backend
.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

### Common Issues & Solutions

**Problem:** Frontend shows CORS errors
**Solution:**
1. Verify backend is actually running: `http://localhost:8000/docs`
2. Check backend logs for request logs when frontend loads
3. Run `.\stop.ps1` to kill orphaned processes before starting

**Problem:** Backend won't start - "Redis connection failed"
**Solution:** This is now fixed. Backend starts without Redis. If you see this error, you have an old version running.

**Problem:** Port 5173 already in use
**Solution:** Run `.\stop.ps1` to kill all Vite processes on ports 5173-5179

**Problem:** Changes to backend code not reflecting
**Solution:** Old uvicorn processes may be running. Use `.\stop.ps1` then restart.

### Testing Checklist

After starting servers, verify:
- [ ] Backend API docs accessible: `http://localhost:8000/docs`
- [ ] Backend root endpoint works: `http://localhost:8000/`
- [ ] Backend health check shows Neo4j healthy: `http://localhost:8000/health`
- [ ] Frontend loads: `http://localhost:5173`
- [ ] Frontend console shows no CORS errors
- [ ] Backend terminal shows request logs from frontend
- [ ] Activities endpoint returns data: `http://localhost:8000/api/v1/activities`

### Dependencies

**Required for Backend:**
- Neo4j running on port 7687
- Python venv with uvicorn installed

**Optional for Backend:**
- Redis on port 6379 (provides caching if available)

**Required for Frontend:**
- Node.js with npm
- Vite dev server

**Required for Scripts:**
- PowerShell (Windows) - Recommended
- Git Bash with Unix tools (Alternative)

---

## Future Improvements

- [ ] Add Redis to Docker Compose for optional local development
- [ ] Create systemd/Windows Service configurations for production
- [ ] Add health check endpoint monitoring in frontend
- [ ] Implement retry logic in frontend API client for transient failures
- [ ] Add environment variable to explicitly disable Redis in production
