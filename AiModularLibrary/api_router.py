import json
import os
from typing import Dict, Any, Optional
from .vendors.openai_client import OpenAIClient
from .vendors.anthropic_client import AnthropicClient
from .vendors.gemini_client import GeminiClient
# from .vendors.deepseek_client import DeepSeekClient


class APIRouter:
    def __init__(self, logger=None):
        # Use the passed logger (from registry)
        self.logger = logger
        
        # Load model configuration
        self.model_config = self._load_model_config()
        
        # Initialize vendor clients
        self.vendor_clients = {
            'OpenAI': OpenAIClient(logger=self.logger.get_child('openai') if self.logger else None),
            'Anthropic': AnthropicClient(logger=self.logger.get_child('anthropic') if self.logger else None),
            'Gemini': GeminiClient(logger=self.logger.get_child('gemini') if self.logger else None),
            # 'DeepSeek': DeepSeekClient(logger=self.logger.get_child('deepseek') if self.logger else None)
        }



    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from JSON file."""
        config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information from config."""
        if model_name not in self.model_config['models']:
            error_msg = f"Model '{model_name}' not found in configuration"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        return self.model_config['models'][model_name]
    
    def _get_vendor_client(self, vendor: str):
        """Get the appropriate vendor client."""
        if vendor not in self.vendor_clients:
            error_msg = f"Vendor '{vendor}' not supported or client not implemented"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        return self.vendor_clients[vendor]
    
    def communicate_with_ai(self, model_name: str, prompt: str,
                           system_message: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: Optional[int] = None,
                           check_for_price: bool = False) -> Dict[str, Any]:
        """
        Main function to communicate with AI systems.
        
        Args:
            model_name: Name of the model (e.g., 'gpt35', 'claude3haiku')
            prompt: User prompt/message
            system_message: Optional system message to set context
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            check_for_price: Whether to calculate and return cost information
        
        Returns:
            Dictionary containing response and optional cost information
        """
        # Get model information
        model_info = self._get_model_info(model_name)
        vendor = model_info['vendor']
        
        # Get the appropriate vendor client
        vendor_client = self._get_vendor_client(vendor)
        
        # Route the request to the vendor client
        # Note: Different vendors might have different parameter names
        # We'll use a common interface but adapt as needed
        try:
            if vendor == 'OpenAI':
                result = vendor_client.call_api(
                    model_name=model_name,
                    prompt=prompt,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    calculate_cost=check_for_price
                )
                return result
            elif vendor == 'Anthropic':
                result = vendor_client.call_api(
                    model_name=model_name,
                    prompt=prompt,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    calculate_cost=check_for_price
                )
                return result
            elif vendor == 'Gemini':
                result = vendor_client.call_api(
                    model_name=model_name,
                    prompt=prompt,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    calculate_cost=check_for_price
                )
                return result
            # elif vendor == 'DeepSeek':
            #     return vendor_client.call_api(
            #         model_name=model_name,
            #         prompt=prompt,
            #         system_message=system_message,
            #         temperature=temperature,
            #         max_tokens=max_tokens,
            #         calculate_cost=check_for_price
            #     )
            else:
                error_msg = f"Vendor '{vendor}' not yet implemented"
                if self.logger:
                    self.logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            error_msg = f"API call failed for model '{model_name}' (vendor: {vendor}): {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
                self.logger.exception("Full exception details")
            raise Exception(error_msg)
    
    def list_available_models(self) -> Dict[str, str]:
        """
        Get a list of all available models grouped by vendor.
        
        Returns:
            Dictionary with vendor as key and list of model names as value
        """
        models_by_vendor = {}
        
        for model_name, model_info in self.model_config['models'].items():
            vendor = model_info['vendor']
            if vendor not in models_by_vendor:
                models_by_vendor[vendor] = []
            models_by_vendor[vendor].append(model_name)
        
        return models_by_vendor
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Dictionary with model information
        """
        return self._get_model_info(model_name) 