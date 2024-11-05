import sys
from classes.YouTubeSummarizer import YouTubeSummarizer
from classes.MultiTools import MultiToolsAgent
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Create CLI menu
    while True:
        print("\nWelcome to AI Agents Experiments!")
        print("1. YouTube Video Summarizer")
        print("2. Multi-Tool Assistant") 
        print("3. Exit")
        choice = input("\nSelect an option (1-3): ")
        
        try:
            if choice == "1":
                url = input("Enter YouTube video URL: ")
                # YouTube_Link = "https://www.youtube.com/watch?v=uh9A4LvuGHM"
                summarizer = YouTubeSummarizer()
                response = summarizer.process_video(url)
                print(f"LLM Response: \n{response}\n\n")
                
            elif choice == "2":
                question = input("Enter your question: ")

                multi_tools = MultiToolsAgent()
                response = multi_tools.ask(question)
                print(f"LLM Response: \n{response}\n\n")
                
            elif choice == "3":
                print("Goodbye!")
                sys.exit(0)
                
            else:
                print("Invalid choice")
                continue
                
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()