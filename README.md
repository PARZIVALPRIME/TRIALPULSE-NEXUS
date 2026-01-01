# TrialPulse Nexus 10X

**Clinical Trial Intelligence Platform**

57,974 Patients | 3,401 Sites | 23 Studies | AI-Powered

## Quick Start (For Your Friend)

### Step 1: Clone Repository
```bash
git clone https://github.com/PARZIVALPRIME/ff.git
cd ff
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Dashboard
```bash
streamlit run dashboard/app.py
```

Open browser at: **http://localhost:8501**

---

## Project Structure

```
trialpulse_nexus/
├── dashboard/           # Streamlit UI
│   ├── app.py          # Main entry point
│   └── pages/          # 6 role-based dashboards
├── src/
│   ├── agents/         # 6 AI agents
│   ├── analytics/      # DQI, Cascade, Clean Patient
│   ├── collaboration/  # Issue Registry, Workspaces
│   ├── generation/     # Report generators
│   ├── governance/     # Audit, Rules Engine
│   ├── knowledge/      # RAG, Embeddings
│   ├── ml/             # ML models
│   └── orchestration/  # Pipeline, Testing
└── data/
    └── raw/            # Input Excel files
```

## Available Dashboards

1. **CRA View** - Field monitoring
2. **DM Hub** - Data management
3. **Safety View** - SAE surveillance  
4. **Study Lead** - Command center
5. **Site Portal** - Site performance
6. **Coder View** - Medical coding

## Test Individual Components

```bash
# Test agents
python src/run_supervisor_test.py

# Test collaboration
python src/run_issue_registry_test.py
python src/run_team_workspaces_test.py

# Test reports
python src/run_report_generators_test.py
```

## Requirements
- Python 3.10+
- 8GB RAM minimum
