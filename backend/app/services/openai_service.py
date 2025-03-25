from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict, Any
import time
from app.config import get_settings
settings = get_settings()
class OpenAIService:
    """Service for Azure OpenAI API integration using Azure AI Inference SDK."""
    
    def __init__(self):
        """Initialize the Azure OpenAI service with settings."""
        self.api_key = settings.openai_api_key
        self.model_name = settings.model_name
        self.azure_endpoint = "https://bpragtest.openai.azure.com/openai/deployments/gpt-4o-mini"
        # self.azure_endpoint = 'https://' + settings.azure_endpoint
        self.temperature = 1.0
        self.max_tokens = 4096
        
        # Initialize the client
        self.client = ChatCompletionsClient(
            endpoint=self.azure_endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
    
    def generate_response(self, query: str, context: List[str] = None) -> Dict[str, Any]:
        """
        Generate a response using Azure AI Inference SDK with RAG context.
        
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
        
        # Create the prompt
        system_content = "You are a helpful assistant."
        user_content = f"""Answer the following question based on the provided context. 
If the context doesn't contain relevant information, just say you don't know 
but provide general information if possible.

Context:
{formatted_context}

Question: {query}

Answer:"""
        
        try:
            # Use the client's complete method
            response = self.client.complete(
                messages=[
                    SystemMessage(content=system_content),
                    UserMessage(content=user_content)
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1.0,
                model=self.model_name
            )
            
            processing_time = time.time() - start_time
            
            # Extract text from response
            response_text = response.choices[0].message.content
            
            return {
                "response": response_text,
                "processing_time": processing_time,
                "model": self.model_name,
                "tokens_used": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "success": True
            }
        except Exception as e:
            # Capture and handle potential errors
            error_message = str(e)
            
            return {
                "response": f"Error generating response: {error_message}",
                "processing_time": time.time() - start_time,
                "model": self.model_name,
                "tokens_used": None,
                "success": False
            }


# Example usage
if __name__ == "__main__":
    service = OpenAIService()
    result = service.generate_response("I am going to Paris, what should I see?")
    print(result["response"])