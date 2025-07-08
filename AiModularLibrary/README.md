# AiModularLibrary

A modular Python library for communicating with multiple AI vendors through a unified interface. Supports OpenAI, Anthropic, and Google Gemini with automatic routing, cost calculation, and response validation.

## Features

- **Multi-Vendor Support**: OpenAI, Anthropic, and Google Gemini
- **Unified API**: Single interface for all vendors
- **Cost Calculation**: Automatic cost tracking for all API calls
- **Response Validation**: Advanced validation system with consistency checks
- **Flexible Configuration**: JSON-based model configuration
- **Comprehensive Logging**: Integrated logging system with the Logger library
- **Error Handling**: Robust error handling and retry mechanisms

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AiModularLibrary
   ```

2. **Install dependencies**:
   ```bash
   pip install -r AiModularLibrary/requirements.txt
   ```

3. **Set up API keys**:
   Create a `secrets.env` file in the main directory with your API keys:
   ```env
   OPENAI_KEY=your_openai_api_key_here
   ANTHROPIC_KEY=your_anthropic_api_key_here
   GEMINI_KEY=your_gemini_api_key_here
   ```

## Quick Start

### Simple API Call

```python
from AiModularLibrary import ai_call_simple

# Call any model directly
result = ai_call_simple('gpt41', 'Hello, how are you?', check_for_price=True)
print(result['response'])
print(f"Cost: ${result['cost']:.6f}")
```

### Validated API Call

```python
from AiModularLibrary import ai_call_checked

# Get validated response with automatic retry and validation
result = ai_call_checked('OpenAI', 'What is the capital of France?', check_for_price=True)
print(result['response'])
print(f"Total cost: ${result['cost']:.6f}")
print(f"Attempts: {result['attempts']}")
```

## Supported Models

### OpenAI Models
- `gpt41` - GPT-4.1 (high quality)
- `gpt41mini` - GPT-4.1 Mini (standard)
- `gpt41nano` - GPT-4.1 Nano (budget)
- `o3` - O3 (reasoning)
- `o4mini` - O4 Mini (reasoning)

### Anthropic Models
- `claude3haiku` - Claude 3.5 Haiku (budget)
- `claude4sonnet` - Claude Sonnet 4 (standard)
- `claude4opus` - Claude Opus 4 (high quality)

### Gemini Models
- `gemini25pro` - Gemini 2.5 Pro (reasoning)
- `gemini25flash` - Gemini 2.5 Flash (standard)
- `gemini25flashlite` - Gemini 2.5 Flash Lite (budget)
- `gemini20flash` - Gemini 2.0 Flash (standard)
- `gemini20flashlite` - Gemini 2.0 Flash Lite (budget)

## Configuration

The library uses `model_config.json` to define models, pricing, and capabilities. Each model includes:

- `vendor`: AI provider (OpenAI, Anthropic, Gemini)
- `vendor_model_id`: Actual model identifier for the API
- `price_per_input_tokens`: Cost per input token
- `price_per_output_tokens`: Cost per output token
- `tokens_provided`: Whether the API provides token counts
- `type`: Model category (high, budget, reasoning, none)

## Advanced Usage

### Custom System Messages

```python
result = ai_call_simple(
    'claude4opus', 
    'Explain quantum computing',
    system_message='You are a helpful physics professor.',
    check_for_price=True
)
```

### Temperature Control

```python
# Use the API router directly for more control
from AiModularLibrary.api_router import APIRouter

router = APIRouter()
result = router.communicate_with_ai(
    model_name='gpt41',
    prompt='Write a creative story',
    temperature=0.9,
    max_tokens=500,
    check_for_price=True
)
```

### List Available Models

```python
from AiModularLibrary.api_router import APIRouter

router = APIRouter()
models = router.list_available_models()
print(models)
# Output: {'OpenAI': ['gpt41', 'gpt41mini', ...], 'Anthropic': [...], 'Gemini': [...]}
```

## Validation System

The `ai_call_checked` function implements a robust validation system:

1. **Consistency Check**: Gets 3 responses from a high-quality model
2. **Validation**: Uses a budget model to validate the responses
3. **Retry Logic**: Up to 5 attempts if validation fails
4. **Cost Aggregation**: Tracks total cost across all calls

This ensures high-quality, consistent responses while managing costs effectively.

## Logging

The library integrates with the Logger system for comprehensive logging:

```python
from Logger import create_logger, LogLevel, register_logger, get_logger

# Initialize logger
logger = create_logger("ai_modular_library", LogLevel.INFO)
register_logger("ai_modular_library", logger)

# Use the library - logging is automatic
result = ai_call_simple('gemini25flash', 'Hello!')
```

## Error Handling

The library provides detailed error messages and handles:
- Missing API keys
- Invalid model names
- API rate limits
- Network errors
- Validation failures

## Cost Management

All API calls can include cost calculation:

```python
result = ai_call_simple('claude3haiku', 'Hello', check_for_price=True)
if 'cost' in result:
    print(f"API call cost: ${result['cost']:.6f}")
```

## Examples

See `example.py` for complete usage examples including:
- Simple API calls
- Validated API calls
- Cost tracking
- Error handling

## Requirements

- Python 3.8+
- Required packages (see `requirements.txt`):
  - `git+https://github.com/RobinJue/Logger.git`
  - `python-dotenv`
  - `openai`
  - `anthropic`
  - `google-generativeai`

## File Structure

```
AiModularLibrary/
├── __init__.py              # Main exports
├── main.py                  # Core functions (ai_call_simple, ai_call_checked)
├── api_router.py            # Vendor routing and communication
├── model_config.json        # Model definitions and pricing
├── requirements.txt         # Dependencies
├── vendors/
│   ├── __init__.py
│   ├── openai_client.py     # OpenAI API client
│   ├── anthropic_client.py  # Anthropic API client
│   └── gemini_client.py     # Gemini API client
└── README.md               # This file
```

## Contributing

To add a new vendor:
1. Create a new client in `vendors/`
2. Update `api_router.py` to include the new vendor
3. Add models to `model_config.json`
4. Update this README

## License

[Add your license information here] 