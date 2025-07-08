import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import openai
from openai import OpenAI


class OpenAIClient:
    def __init__(self, logger=None):
        # Initialize logger
        self.logger = logger
        
        # Load environment variables from secrets.env
        # Path: vendors/ -> AiModularLibrary/ -> main directory/
        secrets_path = os.path.join(os.path.dirname(__file__), '..', '..', 'secrets.env')
        load_dotenv(secrets_path)
        
        # Initialize OpenAI client with API key
        api_key = os.getenv('OPENAI_KEY')
        if not api_key:
            error_msg = "OPENAI_KEY not found in secrets.env"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.logger and not api_key.strip():
            self.logger.warning("OPENAI_KEY appears to be empty or whitespace only")
        
        self.client = OpenAI(api_key=api_key)
        
        # Load model configuration
        self.model_config = self._load_model_config()
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from JSON file."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'model_config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information from config."""
        if model_name not in self.model_config['models']:
            error_msg = f"Model '{model_name}' not found in configuration"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        model_info = self.model_config['models'][model_name]
        if model_info['vendor'] != 'OpenAI':
            error_msg = f"Model '{model_name}' is not an OpenAI model"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        return model_info
    
    def _calculate_cost(self, model_info: Dict[str, Any], 
                       input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost based on token usage and model pricing."""
        input_cost = input_tokens * model_info['price_per_input_tokens']
        output_cost = output_tokens * model_info['price_per_output_tokens']
        total_cost = input_cost + output_cost
        
        return total_cost
    
    def call_api(self, model_name: str, prompt: str, 
                 system_message: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 calculate_cost: bool = False) -> Dict[str, Any]:
        """
        Call OpenAI API with the specified parameters.
        
        Args:
            model_name: Name of the model (e.g., 'gpt35', 'gpt4')
            prompt: User prompt
            system_message: Optional system message
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            calculate_cost: Whether to calculate and return cost information
        
        Returns:
            Dictionary containing API response and optional cost information
        """
        # Get model information
        model_info = self._get_model_info(model_name)
        vendor_model_id = model_info['vendor_model_id']
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare API parameters
        api_params = {
            "model": vendor_model_id,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            api_params["max_tokens"] = max_tokens
        
        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(**api_params)
            
            # Extract response content
            result = {
                "response": response.choices[0].message.content,
                "model_used": vendor_model_id,
                "finish_reason": response.choices[0].finish_reason
            }
            
            # Calculate cost if requested
            if calculate_cost:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_cost = self._calculate_cost(model_info, input_tokens, output_tokens)
                result["cost"] = total_cost
                
                if self.logger and total_cost > 0.01:  # Warn for expensive calls
                    self.logger.warning(f"High cost API call: ${total_cost:.6f} for {input_tokens + output_tokens} tokens")
            
            return result
            
        except Exception as e:
            error_msg = f"OpenAI API call failed: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
                self.logger.exception("Full exception details")
            raise Exception(error_msg) 