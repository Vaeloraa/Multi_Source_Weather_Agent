"""清除会话中的 study_case 决策，保留对话历史。"""
import json
from pathlib import Path

session_file = Path("d:/Profile/专业实习/sevir-agent-demo/data/runtime/agent_sessions/_latest_session.json")

if session_file.exists():
    with open(session_file, 'r', encoding='utf-8') as f:
        session = json.load(f)
    
    # 清除 study_case 决策，但保留对话历史
    if 'study_case' in session:
        old_decisions = session['study_case'].get('decisions', {})
        print(f"Clearing decisions: {old_decisions}")
        session['study_case']['decisions'] = {}
        session['study_case']['records'] = []
    
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    
    print(f"Cleared session decisions, keeping {len(session.get('turns', []))} conversation turns")
else:
    print("Session file not found")
