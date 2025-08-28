from youtube_transcript_api import YouTubeTranscriptApi

video_id = "nb_fFj_0rq8"  # Example video ID
ytt_api = YouTubeTranscriptApi()
transcript = ytt_api.list(video_id)
print(transcript)

fetched_transcript = ytt_api.fetch(video_id, languages=["en-US"])

for snippet in fetched_transcript:
    print(snippet.text)