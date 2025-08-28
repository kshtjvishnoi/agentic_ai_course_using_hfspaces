from agent.tools.media_tools import *

def we_lookup_tool():
    state = {"question": "Summarize the audio file in a concise manner."}
    answer = asr_tool(state, file_path="C:/Users/joyan/agentic_certification_project/certification_course_agentic_ai/agent/data/Monologue.mp3")
    print(answer)
    return answer

if __name__ == "__main__":
    we_lookup_tool()

# from openai import OpenAI

# client = OpenAI()  # needs OPENAI_API_KEY in your env

# audio_path = r"C:/Users/joyan/agentic_certification_project/certification_course_agentic_ai/agent/data/Monologue.mp3"

# with open(audio_path, "rb") as f:
#     # "whisper-1" is the classic; some accounts also have "gpt-4o-transcribe"
#     transcript = client.audio.transcriptions.create(
#         model="whisper-1",
#         file=f
#     )

# print(transcript.text)