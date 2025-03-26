from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
import time
import openai

from app.config import get_settings

settings = get_settings()

class OpenAIService:
    """Service for OpenAI API integration."""
    
    def __init__(self):
        """Initialize the OpenAI service with API key from settings."""
        self.api_key = settings.openai_api_key
        self.model_name = settings.model_name
        self.temperature = 0.7
        self.max_tokens = 5000
        
        # Set OpenAI API key directly
        openai.api_key = self.api_key
    
    def generate_response(self, query: str, context: List[str] = None) -> Dict[str, Any]:
        """
        Generate a response using OpenAI's API with RAG context.
        
        Args:
            query (str): The user's query
            context (List[str], optional): Context information from retrieved chunks
            
        Returns:
            Dict[str, Any]: Response object with generated text and metadata
        """
        start_time = time.time()
        
        if not context:
            context = ["No additional context available."]
        
        # Format context for the prompt
        formatted_context = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        # Create prompt text
        prompt_text = f"""
        Answer the following question based on the provided context.
        If the context doesn't contain relevant information, just say you don't know
        but provide general information if possible.

        Context:
        {formatted_context}

        Question: {query}

        Answer:
        """
        
        # Use OpenAI API directly
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content
            
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "processing_time": time.time() - start_time,
                "model": self.model_name,
                "tokens_used": None,
                "success": False
            }
        
        processing_time = time.time() - start_time
        
        return {
            "response": result,
            "processing_time": processing_time,
            "model": self.model_name,
            "tokens_used": None,
            "success": True
        }