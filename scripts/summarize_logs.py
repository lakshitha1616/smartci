
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load API key from the project root
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# Read the build log
log_file = ROOT / "build.log"

if not log_file.exists():
    print("Error: build.log file not found")
    sys.exit(1)

logs = log_file.read_text(encoding="utf-8", errors="replace")

if not logs.strip():
    print("Error: build.log is empty")
    sys.exit(1)

prompt = f"""
Analyze the following CI/CD build logs.

Provide:
1. Build status (success or failure)
2. Summary of important events
3. Errors and warnings
4. Possible root causes
5. Recommended fixes

Do not invent errors or causes not supported by the logs.

Build Logs:
{logs[:20000]}
"""



response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt
)


summary = response.text or "No summary returned."

print("\n===== AI BUILD LOG SUMMARY =====\n")
print(summary)

(ROOT / "build_summary.txt").write_text(
    summary, encoding="utf-8"
)

print("\nSummary saved to build_summary.txt")
