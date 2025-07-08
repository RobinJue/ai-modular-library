from AiModularLibrary import ai_call_simple, ai_call_checked
from Logger import create_logger, LogLevel, register_logger, get_logger


# The main functions use the logger from the registry system
# This ensures consistent logging across the entire application


# Initialize global logger once
def initialize_logger():
    """Initialize and register the global logger."""
    logger = create_logger("ai_modular_library", LogLevel.INFO)
    register_logger("ai_modular_library", logger)
    return logger


def test_gemini25flash():
    """
    Test function that sends a prompt to Gemini 2.5 Flash model.
    """
    # Get the global logger
    logger = get_logger("ai_modular_library")
    if not logger:
        logger = initialize_logger()
    
    logger.info("Starting AI test with Gemini 2.5 Flash")
    
    # Test prompt
    test_prompt = "Hello! Can you tell me a short joke?"
    
    logger.info("Testing Gemini 2.5 Flash model...")
    logger.info(f"Test prompt: {test_prompt}")
    
    try:
        # Call the AI with cost calculation enabled
        logger.info("Calling AI with cost calculation enabled")
        result = ai_call_simple('gemini25flash', test_prompt, check_for_price=True)
        
        # Log results
        logger.info("AI Response:")
        logger.info(result['response'])
        logger.info("AI response received successfully")
        
        # Log cost information
        if 'cost' in result:
            logger.info(f"Total cost: ${result['cost']:.6f}")
        else:
            logger.warning("No cost information returned")
        
        logger.info(f"Model used: {result['model_used']}")
        logger.info(f"Finish reason: {result['finish_reason']}")
        
    except Exception as e:
        error_msg = f"Error during AI call: {e}"
        print(f"Error: {e}")
        logger.error(error_msg)
        logger.exception("Full exception details")


def test_ai_call_checked():
    """
    Test function that uses the validated AI call with Gemini.
    """
    # Get the global logger
    logger = get_logger("ai_modular_library")
    if not logger:
        logger = initialize_logger()
    
    logger.info("Starting validated AI test with Gemini")
    
    # Test prompt
    test_prompt = "What is the capital of France?"
    
    logger.info("Testing validated AI call with Gemini...")
    logger.info(f"Test prompt: {test_prompt}")
    
    try:
        # Call the validated AI with cost calculation enabled
        logger.info("Calling validated AI with cost calculation enabled")
        result = ai_call_checked('Gemini', test_prompt, check_for_price=True)
        
        # Log results
        logger.info("Validated AI Response:")
        logger.info(result['response'])
        logger.info("Validated AI response received successfully")
        
        # Log cost information
        if 'cost' in result:
            logger.info(f"Total cost: ${result['cost']:.6f}")
        else:
            logger.warning("No cost information returned")
        
        logger.info(f"Model used: {result['model_used']}")
        logger.info(f"Finish reason: {result['finish_reason']}")
        logger.info(f"Attempts: {result['attempts']}")
        
    except Exception as e:
        error_msg = f"Error during validated AI call: {e}"
        logger.error(error_msg)
        logger.exception("Full exception details")


if __name__ == "__main__":
    # Initialize logger once at startup
    initialize_logger()
    test_gemini25flash()
    logger = get_logger("ai_modular_library")
    logger.info("=" * 50)
    test_ai_call_checked() 