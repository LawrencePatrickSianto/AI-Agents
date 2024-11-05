from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from utils.prompts import analyze_prompt


class YouTubeSummarizer:
    """A class that extracts and summarizes YouTube video transcripts using AI"""

    def __init__(self):
        """Initialize YouTubeSummarizer with default number of key points"""
        self.model_name = "llama-3.2-3b-preview"
        self.top_n_points = 5
        print(f"Using AI Model: {self.model_name}\n")

    def get_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL
        
        Args:
            url (str): Full YouTube video URL (supports youtube.com and youtu.be formats)
            
        Returns:
            str: Extracted video ID
            
        Raises:
            ValueError: If URL is not a recognized YouTube URL format
        """
        if "youtu.be" in url:
            # Handle shortened URLs (e.g., https://youtu.be/VIDEO_ID)
            return url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            # Handle full URLs (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
            return url.split("v=")[1].split("&")[0]
        else:
            raise ValueError("URL must be a valid YouTube URL (youtube.com or youtu.be format)")

    def get_transcript(self, video_id: str) -> str:
        """Fetch and combine transcript segments from YouTube video
        
        Args:
            video_id (str): YouTube video identifier
            
        Returns:
            str: Complete video transcript as continuous text
            
        Raises:
            ValueError: If video_id is None or empty
            Exception: If transcript cannot be retrieved
        """
        if not video_id:
            raise ValueError("Video ID is required")
        
        try:
            transcript_segments = YouTubeTranscriptApi.get_transcript(video_id)
            full_transcript = " ".join([segment["text"] for segment in transcript_segments])
            return full_transcript
        except Exception as e:
            raise Exception(f"Failed to retrieve transcript: {str(e)}")
    
    def get_prompt_tokens(self, transcript: str, prompt: PromptTemplate) -> None:
        """Analyze token count of the formatted summarization prompt
        
        Args:
            transcript (str): Video transcript to analyze
            prompt (PromptTemplate): Template to use for summarization
        """
        formatted_prompt = prompt.format(
            YouTube_Transcript=transcript, 
            top_n=self.top_n_points
        )
        analyze_prompt(formatted_prompt)
    
    def create_prompt(self) -> PromptTemplate:
        """Create structured prompt template for video summarization
        
        Returns:
            PromptTemplate: Template with instructions for AI summarization
        """
        template = """
        Summarize the following YouTube video transcript:

        {YouTube_Transcript}

        Please provide:
        1. A brief overview of the video's main topic or theme
        2. Identify the speakers.
        3. The top {top_n} most important points or insights presented, with a short explanation for each
        4. Any notable quotes or statements made by the speaker(s)
        5. Key takeaways or conclusions from the video

        Your response should be well-structured and easy to read
        """

        return PromptTemplate(
            template=template, 
            input_variables=["YouTube_Transcript", "top_n"]
        )
    
    def summarize(self, transcript: str) -> str:
        """Generate AI summary of video transcript
        
        Args:
            transcript (str): Complete video transcript text
            
        Returns:
            str: Structured summary of the video content
            
        Raises:
            ValueError: If transcript is empty or None
            Exception: If AI processing fails
        """
        if not transcript:
            raise ValueError("Transcript is required for summarization")

        # Prepare prompt and analyze token usage
        prompt = self.create_prompt()
        self.get_prompt_tokens(transcript, prompt)

        # Configure AI model
        model = ChatGroq(
            model=self.model_name,
            temperature=0.5,  # Balance between creativity and consistency
            max_tokens=None,  # Allow dynamic response length
            timeout=None,
            max_retries=2,
        )

        try:
            # Create and execute processing pipeline
            chain = (
                {
                    "YouTube_Transcript": RunnablePassthrough(),
                    "top_n": lambda _: self.top_n_points
                }
                | prompt
                | model
                | StrOutputParser()
            )

            return chain.invoke(transcript)
        except Exception as e:
            raise Exception(f"AI summarization failed: {str(e)}")

    def process_video(self, url: str) -> str:
        """Complete pipeline to summarize a YouTube video from URL
        
        Args:
            url (str): YouTube video URL to process
            
        Returns:
            str: AI-generated structured summary
            
        Raises:
            ValueError: If URL is invalid
            Exception: If any processing step fails
        """
        video_id = self.get_video_id(url)
        transcript = self.get_transcript(video_id)
        return self.summarize(transcript)
