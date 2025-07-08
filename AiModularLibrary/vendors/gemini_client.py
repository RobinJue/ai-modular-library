import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai


class GeminiClient:
    def __init__(self, logger=None):
        # Initialize logger
        self.logger = logger
        
        # Load environment variables from secrets.env
        # Path: vendors/ -> AiModularLibrary/ -> main directory/
        secrets_path = os.path.join(os.path.dirname(__file__), '..', '..', 'secrets.env')
        load_dotenv(secrets_path)
        
        # Initialize Gemini client with API key
        api_key = os.getenv('GEMINI_KEY')
        if not api_key:
            error_msg = "GEMINI_KEY not found in secrets.env"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.logger and not api_key.strip():
            self.logger.warning("GEMINI_KEY appears to be empty or whitespace only")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
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
        if model_info['vendor'] != 'Gemini':
            error_msg = f"Model '{model_name}' is not a Gemini model"
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
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple estimation: ~4 characters per token for English text
        return len(text) // 4
    
    def call_api(self, model_name: str, prompt: str, 
                 system_message: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 calculate_cost: bool = False) -> Dict[str, Any]:
        """
        Call Gemini API with the specified parameters.
        
        Args:
            model_name: Name of the model (e.g., 'gemini25pro', 'gemini25flash')
            prompt: User prompt
            system_message: Optional system message
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            calculate_cost: Whether to calculate and return cost information
        
        Returns:
            Dictionary containing API response and optional cost information
        """
        # Get model information
        model_info = self._get_model_info(model_name)
        vendor_model_id = model_info['vendor_model_id']
        
        # Prepare the full prompt
        full_prompt = prompt
        if system_message:
            full_prompt = f"[System: {system_message}]\n\n{prompt}"
        
        # Prepare generation config
        generation_config = {
            "temperature": temperature,
        }
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        # Call Gemini API
        try:
            # Create model instance
            model = genai.GenerativeModel(
                model_name=vendor_model_id,
                generation_config=generation_config
            )
            
            # Generate response
            response = model.generate_content(full_prompt)
            
            # Extract response content
            result = {
                "response": response.text,
                "model_used": vendor_model_id,
                "finish_reason": "stop"  # Gemini doesn't provide detailed finish reasons
            }
            
            # Calculate cost if requested
            if calculate_cost:
                # Estimate token usage since Gemini doesn't provide exact counts
                input_tokens = self._estimate_tokens(full_prompt)
                output_tokens = self._estimate_tokens(response.text)
                total_cost = self._calculate_cost(model_info, input_tokens, output_tokens)
                result["cost"] = total_cost
                
                if self.logger and total_cost > 0.01:  # Warn for expensive calls
                    self.logger.warning(f"High cost API call: ${total_cost:.6f} for {input_tokens + output_tokens} tokens")
            
            return result
            
        except Exception as e:
            error_msg = f"Gemini API call failed: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
                self.logger.exception("Full exception details")
            raise Exception(error_msg) 