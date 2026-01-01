"""Check actual tool names in registry - FIXED"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.tools import ToolRegistry

registry = ToolRegistry()
tools = registry.list_tools()

print("AVAILABLE TOOLS:")
print("=" * 50)
if isinstance(tools, dict):
    for name, info in tools.items():
        print(f"  - {name}: {info.get('description', '')[:60]}...")
elif isinstance(tools, list):
    for tool in tools:
        if isinstance(tool, dict):
            print(f"  - {tool.get('name', tool)}")
        else:
            print(f"  - {tool}")