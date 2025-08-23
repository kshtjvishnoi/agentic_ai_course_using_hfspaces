# quick test (text-only)
from agent.state import State
from agent.tools.openai_engine import openai_answer_tool

s = {
  "task_id": "t1",
  "question": "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer.",
  "file_name": "",
}
print(openai_answer_tool(s))  # expect: 5 (just the number)