from typing import Dict, Any, Optional
from .api_router import APIRouter
from Logger import get_logger


def ai_call_simple(model_name: str, prompt: str, check_for_price: bool = False) -> Dict[str, Any]:
    """
    Simple function to communicate with AI systems.
    
    This function provides a straightforward interface to call any AI model directly.
    It routes the request through the APIRouter to the appropriate vendor client.
    
    Args:
        model_name: Name of the model (e.g., 'gpt35', 'claude3haiku', 'gemini15flash')
        prompt: User prompt/message to send to the AI
        check_for_price: Whether to calculate and return cost information (default: False)
    
    Returns:
        Dictionary containing:
            - prompt: Original user prompt
            - response: AI-generated response
            - model_used: Actual model identifier used by the vendor
            - finish_reason: Why the response finished (e.g., 'stop', 'length')
            - cost: Total cost in dollars (only if check_for_price=True)
    
    Raises:
        ValueError: If model_name is not found in configuration
        Exception: If API call fails or vendor is not supported
    """
    # Get logger from registry for consistent logging across the application
    logger = get_logger("ai_modular_library")
    
    # Create API router instance to handle vendor-specific routing
    router = APIRouter(logger=logger)
    
    # Log the start of the AI call
    logger.info(f"Starting AI call with model: {model_name}")
    
    # Make the API call through the router
    result = router.communicate_with_ai(
        model_name=model_name,
        prompt=prompt,
        check_for_price=check_for_price
    )
    
    # Log successful completion
    logger.info(f"AI call completed successfully with model: {result['model_used']}")
    if check_for_price and 'cost' in result:
        logger.info(f"Total cost: ${result['cost']:.6f}")
    
    # Add original prompt to result for consistency and debugging
    result["prompt"] = prompt
    return result


def ai_call_checked(vendor: str, prompt: str, check_for_price: bool = False) -> Dict[str, Any]:
    """
    Validated AI call that uses a high-quality model and validates responses with a budget model.
    
    This function implements a robust validation system that:
    1. Gets 3 responses from a high-quality model to ensure consistency
    2. Validates the responses using a budget model to check accuracy
    3. Retries up to 5 times if validation fails
    4. Aggregates costs from all API calls (3 high model + 1 budget model per attempt)
    
    The validation process helps ensure high-quality, consistent responses by:
    - Checking if all 3 high-model responses are identical
    - Using a budget model to validate the correctness of the responses
    - Only returning responses that pass both consistency and validation checks
    
    Args:
        vendor: Vendor to use ('OpenAI', 'Anthropic', 'Gemini')
        prompt: User prompt/message to send to the AI
        check_for_price: Whether to calculate and return cost information (default: False)
    
    Returns:
        Dictionary containing:
            - prompt: Original user prompt
            - response: Validated AI response or error message
            - model_used: High model name with validation info
            - finish_reason: 'validated' or 'validation_failed'
            - attempts: Number of attempts made (1-5)
            - cost: Total cost from all API calls (only if check_for_price=True)
    
    Raises:
        ValueError: If vendor is not supported or no high/budget models found
        Exception: If all 5 validation attempts fail
    """
    # Get logger from registry for consistent logging across the application
    logger = get_logger("ai_modular_library")
    
    # Validate that the requested vendor is supported
    # Only vendors with both high and budget models are allowed
    allowed_vendors = ['OpenAI', 'Anthropic', 'Gemini']
    if vendor not in allowed_vendors:
        error_msg = f"Vendor '{vendor}' not allowed. Allowed vendors: {allowed_vendors}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Create API router instance to handle vendor-specific routing and model configuration
    router = APIRouter(logger=logger)
    
    # Load the complete model configuration to find appropriate models for this vendor
    # The configuration contains all available models with their types (high, budget, reasoning, none)
    model_config_data = router.model_config
    
    # Find the appropriate high-quality and budget models for this vendor
    # High models are used for generating the main responses (better quality, higher cost)
    # Budget models are used for validation to keep costs low while ensuring accuracy
    high_model = None
    budget_model = None
    
    # Iterate through all models in the configuration to find models matching the vendor
    # and having the correct type classification
    for model_name, model_info in model_config_data['models'].items():
        if model_info['vendor'] == vendor:
            if model_info['type'] == 'high':
                high_model = model_name
            elif model_info['type'] == 'budget':
                budget_model = model_name
    
    # Ensure a high-quality model is available for generating responses
    # High models are essential for the validation process
    if not high_model:
        error_msg = f"No high model found for vendor: {vendor}. High models are required for response generation."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Ensure a budget model is available for validation
    # Budget models are used to validate responses at a lower cost
    if not budget_model:
        error_msg = f"No budget model found for vendor: {vendor}. Budget models are required for response validation."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Log the start of the validation process with selected models
    logger.info(f"Starting validated AI call for vendor: {vendor}")
    logger.info(f"Using high model: {high_model}, budget model: {budget_model}")
    
    # Initialize total cost tracking for all API calls across all attempts
    total_cost = 0.0
    
    # Main validation loop: try up to 5 times to get a validated response
    # This retry mechanism handles cases where responses are inconsistent or validation fails
    for attempt in range(5):
        logger.info(f"Validation attempt {attempt + 1}/5")
        
        try:
            # Step 1: Get three responses from the high-quality model
            # This ensures consistency by checking if the model gives the same answer multiple times
            responses = []
            logger.info(f"Getting 3 responses from high model for consistency check")
            for i in range(3):
                result = router.communicate_with_ai(
                    model_name=high_model,
                    prompt=prompt,
                    check_for_price=check_for_price
                )
                responses.append(result['response'])
                # Track costs if price calculation is enabled
                if check_for_price and 'cost' in result:
                    total_cost += result['cost']
            
            # Step 2: Check if all three responses are identical
            # Using set() to remove duplicates - if length is 1, all responses are the same
            if len(set(responses)) == 1:
                logger.info("All 3 responses are identical - proceeding to validation")
                
                # Step 3: Create a validation prompt for the budget model
                # This prompt instructs the budget model to act as a validation expert
                # and check if the responses are both identical and valid
                validation_prompt = f"""You are a validation expert. I will give you an original prompt and three AI responses. 
Your task is to check if all three responses are identical and valid.

Original prompt: {prompt}

Response 1: {responses[0]}
Response 2: {responses[1]}
Response 3: {responses[2]}

If all three responses are identical and valid, respond with the correct answer only.
If they are not identical or invalid, respond with exactly: %FALSE%

Your response:"""
                
                # Step 4: Get validation from the budget model
                # This is a cost-effective way to validate the quality of the responses
                logger.info("Validating responses with budget model")
                validation_result = router.communicate_with_ai(
                    model_name=budget_model,
                    prompt=validation_prompt,
                    check_for_price=check_for_price
                )
                
                # Extract the validation response and track costs
                validation_response = validation_result['response'].strip()
                if check_for_price and 'cost' in validation_result:
                    total_cost += validation_result['cost']
                
                # Step 5: Check validation result
                # If the budget model returns "%FALSE%", it means the responses failed validation
                # Otherwise, the budget model returns the validated/corrected response
                if validation_response == "%FALSE%":
                    logger.warning(f"Validation failed on attempt {attempt + 1} - responses deemed invalid")
                    continue
                else:
                    # Validation successful - return the validated response
                    logger.info(f"Validation successful after {attempt + 1} attempt(s)")
                    if check_for_price:
                        logger.info(f"Total cost for all calls: ${total_cost:.6f}")
                    
                    result = {
                        "prompt": prompt,
                        "response": validation_response,
                        "model_used": f"{high_model} (validated by {budget_model})",
                        "finish_reason": "validated",
                        "attempts": attempt + 1
                    }
                    if check_for_price:
                        result["cost"] = total_cost
                    return result
            
            else:
                # If responses are not identical, log and continue to next attempt
                # This ensures we only proceed with validation when we have consistent responses
                logger.warning(f"Responses not identical on attempt {attempt + 1} - retrying")
                continue
                
        except Exception as e:
            # Handle any API errors or unexpected issues during the process
            # This could include network issues, API rate limits, or vendor-specific errors
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt == 4:  # Last attempt - re-raise the exception to prevent silent failures
                raise e
    
    # If we reach here, all 5 attempts have failed
    # This could be due to inconsistent responses, validation failures, or API errors
    # We return a structured error response rather than raising an exception
    error_msg = f"Failed to get validated response after 5 attempts for vendor: {vendor}"
    logger.error(error_msg)
    if check_for_price:
        logger.error(f"Total cost for failed attempts: ${total_cost:.6f}")
    
    # Return error result with all the information about the failed attempts
    # This allows the caller to handle the failure gracefully and see what happened
    result = {
        "prompt": prompt,
        "response": f"ERROR: {error_msg}",
        "model_used": f"{high_model} (validation failed)",
        "finish_reason": "validation_failed",
        "attempts": 5
    }
    
    # Include total cost even for failed attempts if price calculation was requested
    if check_for_price:
        result["cost"] = total_cost
    
    return result 