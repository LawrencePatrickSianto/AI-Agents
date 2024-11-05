from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import Tool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.utilities import PythonREPL
from utils.prompts import analyze_prompt


class MultiToolsAgent:
    """An agent that can use multiple tools (Python REPL and Wikipedia) to answer questions"""
    
    def __init__(self):
        """Initialize the agent with empty tool configurations"""
        self.model_name = "llama-3.2-3b-preview"
        print(f"Using AI Model: {self.model_name}\n")

        self.tools = []
        self.tools_str = ""  # String representation of available tools
        self.tool_names = [] # List of tool names for prompt formatting
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """Set up the available tools for the agent"""
        # Initialize tool instances
        python_repl = Tool(
            name='python repl',
            func=PythonREPL().run,
            description="Useful for executing Python code to answer computational questions"
        )
        
        wikipedia = Tool(
            name='wikipedia',
            func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
            description="Useful for looking up factual information about topics, countries or people"
        )

        # Store configured tools
        self.tools = [python_repl, wikipedia]
        
        # Create formatted strings for prompt templates
        self.tools_str = "".join([f"{t.name}: {t.description}\n" for t in self.tools])
        self.tool_names = [tool.name for tool in self.tools]
        
    def _create_prompt(self) -> PromptTemplate:
        """Create the instruction prompt for the agent"""
        template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of {tool_names}
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {question}
        """

        return PromptTemplate(
            template=template,
            input_variables=["tools", "tool_names", "question"]
        )
    
    def _analyze_prompt_tokens(self, question: str, prompt: PromptTemplate) -> None:
        """Analyze token count of the formatted prompt
        
        Args:
            question (str): The user's question
            prompt (PromptTemplate): The prompt template to analyze
        """
        prompt_text = prompt.format(
            tools=self.tools_str,
            tool_names=self.tool_names,
            question=question
        )
        analyze_prompt(prompt_text)

    def ask(self, question: str) -> str:
        """Process a question using the AI model and available tools
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The AI-generated answer
            
        Raises:
            ValueError: If question is None
            Exception: If there's an error during processing
        """
        if question is None:
            raise ValueError("Question cannot be None")

        # Set up the prompt and model
        prompt = self._create_prompt()
        self._analyze_prompt_tokens(question, prompt)

        model = ChatGroq(
            model=self.model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        try:
            # Create and execute the processing chain
            chain = (
                {
                    "question": RunnablePassthrough(),
                    "tools": lambda _: self.tools_str,
                    "tool_names": lambda _: self.tool_names
                }
                | prompt
                | model
                | StrOutputParser()
            )

            return chain.invoke(question)
            
        except Exception as e:
            raise Exception(f"Error processing question: {str(e)}")
