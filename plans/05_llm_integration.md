# Phase 5: LLM Integration Tests with OpenRouter

## OpenRouter Setup Tests

### API Configuration
- Test API key validation
- Test model selection (Claude-3.5-sonnet)
- Test endpoint configuration
- Test timeout settings
- Test rate limiting
- Test error handling
- Test retry mechanisms
- Test connection pooling
- Test async support
- Test session management

### Model Configuration
- Test model parameters
- Test temperature settings
- Test max tokens
- Test stop sequences
- Test prompt templates
- Test system messages
- Test response formats
- Test streaming settings
- Test context window
- Test cost tracking

### Error Handling
- Test invalid API key
- Test model unavailability
- Test rate limit exceeded
- Test timeout handling
- Test malformed requests
- Test network errors
- Test response validation
- Test fallback strategies
- Test error reporting
- Test recovery procedures

## LLM Interaction Tests

### Basic Operations
```python
class TestLLMBasics(unittest.TestCase):
    def setUp(self):
        self.client = OpenRouterClient(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="anthropic/claude-3-sonnet"
        )
        self.test_prompt = self.get_test_prompt()
        
    def get_test_prompt(self):
        return {
            "system": "You are a helpful AI assistant.",
            "user": "What is the capital of France?"
        }
    
    def test_basic_completion(self):
        response = self.client.complete(self.test_prompt)
        self.assertIsNotNone(response)
        self.assertIn("Paris", response.text)
        
    def test_streaming_completion(self):
        chunks = []
        for chunk in self.client.complete_stream(self.test_prompt):
            chunks.append(chunk)
        self.assertTrue(len(chunks) > 0)
        
    def test_token_counting(self):
        tokens = self.client.count_tokens(self.test_prompt)
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 0)
```

### Advanced Usage
```python
class TestLLMAdvanced(unittest.TestCase):
    def test_context_management(self):
        # Test handling large context
        long_prompt = "..." * 10000  # Large context
        response = self.client.complete_with_context_handling(long_prompt)
        self.assertIsNotNone(response)
        
    def test_batch_processing(self):
        # Test batch request handling
        prompts = [self.get_test_prompt() for _ in range(5)]
        responses = self.client.complete_batch(prompts)
        self.assertEqual(len(responses), len(prompts))
        
    def test_async_operations(self):
        # Test async API usage
        async def run_async_test():
            responses = await asyncio.gather(*[
                self.client.acomplete(self.get_test_prompt())
                for _ in range(3)
            ])
            return responses
            
        responses = asyncio.run(run_async_test())
        self.assertEqual(len(responses), 3)
```

### Error Scenarios
```python
class TestLLMErrors(unittest.TestCase):
    def test_rate_limit_handling(self):
        # Test rate limit recovery
        with self.assertRaises(RateLimitError):
            for _ in range(100):  # Exceed rate limit
                self.client.complete(self.get_test_prompt())
                
        time.sleep(60)  # Wait for rate limit reset
        response = self.client.complete(self.get_test_prompt())
        self.assertIsNotNone(response)
        
    def test_timeout_handling(self):
        # Test timeout recovery
        with self.assertRaises(TimeoutError):
            self.client.complete(
                self.get_test_prompt(),
                timeout=0.001  # Very short timeout
            )
            
        response = self.client.complete(
            self.get_test_prompt(),
            timeout=30  # Normal timeout
        )
        self.assertIsNotNone(response)
        
    def test_invalid_requests(self):
        # Test invalid request handling
        with self.assertRaises(ValidationError):
            self.client.complete({"invalid": "prompt"})
```

## DSPy Integration Tests

### LLM Module Tests
```python
class TestDSPyLLM(unittest.TestCase):
    def setUp(self):
        self.llm = OpenRouterLLM(
            model="anthropic/claude-3-sonnet",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.dspy_module = dspy.Module(self.llm)
        
    def test_basic_module(self):
        # Test basic DSPy module with LLM
        signature = dspy.Signature(
            inputs=["question"],
            outputs=["answer"]
        )
        
        module = dspy.Module(signature)
        response = module("What is 2+2?")
        self.assertIn("4", response.answer)
        
    def test_chain_of_thought(self):
        # Test CoT reasoning
        signature = dspy.Signature(
            inputs=["question"],
            outputs=["reasoning", "answer"]
        )
        
        module = dspy.ChainOfThought(signature)
        response = module("What is the square root of 16?")
        self.assertIn("4", response.answer)
        self.assertNotEqual(response.reasoning, "")
```

### Pipeline Integration
```python
class TestPipelineIntegration(unittest.TestCase):
    def test_llm_preprocessing(self):
        # Test LLM-based data preprocessing
        preprocessor = LLMPreprocessor(self.llm)
        raw_data = "Noisy data..."
        cleaned = preprocessor.clean(raw_data)
        self.assertNotEqual(cleaned, raw_data)
        
    def test_llm_postprocessing(self):
        # Test LLM-based result enhancement
        postprocessor = LLMPostprocessor(self.llm)
        results = ["Result 1", "Result 2"]
        enhanced = postprocessor.enhance(results)
        self.assertEqual(len(enhanced), len(results))
        
    def test_llm_validation(self):
        # Test LLM-based validation
        validator = LLMValidator(self.llm)
        test_output = "Test result"
        validation = validator.validate(test_output)
        self.assertIsInstance(validation["score"], float)
```

## Performance and Cost Tests

### Performance Metrics
- Test response latency
- Test token processing speed
- Test batch processing efficiency
- Test streaming performance
- Test concurrent request handling
- Test memory usage
- Test CPU utilization
- Test network efficiency
- Test caching effectiveness
- Test rate limit optimization

### Cost Tracking
- Test token counting accuracy
- Test cost calculation
- Test budget management
- Test usage monitoring
- Test cost optimization
- Test billing integration
- Test usage reporting
- Test cost alerts
- Test quota management
- Test cost forecasting

## Success Criteria

### Integration Quality
- Seamless OpenRouter API integration
- Reliable model responses
- Proper error handling
- Efficient resource usage

### Performance Targets
- Response time < 2s for basic queries
- Streaming latency < 100ms
- Batch processing efficiency > 0.8
- Rate limit compliance 100%

### Cost Efficiency
- Token usage optimization
- Budget adherence
- Cost tracking accuracy
- Usage within quotas

### Reliability
- 99.9% API availability
- Error recovery > 95%
- Zero data loss
- Consistent performance

## Development Workflow
1. Setup API integration
2. Implement basic operations
3. Add advanced features
4. Optimize performance
5. Add monitoring
6. Document usage

## Tools and Dependencies
- openrouter-py
- pytest-asyncio
- pytest-timeout
- aiohttp
- tenacity
- prometheus-client
- datadog (optional)
- sentry-sdk (optional)
