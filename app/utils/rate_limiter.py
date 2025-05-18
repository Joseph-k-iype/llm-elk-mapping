"""
Enhanced rate limiter for API request throttling.

This module provides classes for rate limiting requests with various strategies:
- Simple rate limiter with fixed rate
- Token bucket rate limiter for allowing short bursts
- Adaptive rate limiter that adjusts based on response times
"""

import time
import asyncio
import logging
import random
from typing import Dict, Any, Optional, Callable, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BaseRateLimiter:
    """Base class for rate limiters."""
    
    async def acquire(self, key: Optional[str] = None, cost: int = 1) -> bool:
        """
        Acquire permission to proceed with a request.
        
        Args:
            key: Optional key for partitioned rate limiting
            cost: Cost of this request (default: 1)
            
        Returns:
            True if request is allowed, False otherwise
        """
        raise NotImplementedError("Subclasses must implement acquire()")
    
    async def wait(self, key: Optional[str] = None, cost: int = 1) -> float:
        """
        Wait until a request is allowed to proceed.
        
        Args:
            key: Optional key for partitioned rate limiting
            cost: Cost of this request (default: 1)
            
        Returns:
            Wait time in seconds
        """
        raise NotImplementedError("Subclasses must implement wait()")
    
    def update_rate(self, rate: float) -> None:
        """
        Update the rate limit.
        
        Args:
            rate: New rate limit (requests per second)
        """
        raise NotImplementedError("Subclasses must implement update_rate()")
    
    def record_response(self, 
                      success: bool, 
                      response_time: float, 
                      status_code: Optional[int] = None,
                      retry_after: Optional[int] = None) -> None:
        """
        Record the result of a request for adaptive rate limiting.
        
        Args:
            success: Whether the request was successful
            response_time: Response time in seconds
            status_code: Optional HTTP status code
            retry_after: Optional retry-after value from response headers
        """
        pass  # Optional in base class

class SimpleRateLimiter(BaseRateLimiter):
    """
    Simple rate limiter that limits requests per second.
    
    This implementation uses a sliding window to track requests.
    """
    
    def __init__(self, rate: float = 10.0, window_size: float = 1.0):
        """
        Initialize the rate limiter.
        
        Args:
            rate: Maximum requests per second
            window_size: Window size in seconds for the sliding window
        """
        self.rate = rate
        self.window_size = window_size
        self.request_times: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
    
    async def acquire(self, key: Optional[str] = None, cost: int = 1) -> bool:
        """
        Check if a request is allowed without waiting.
        
        Args:
            key: Optional key for partitioned rate limiting (default: None means global)
            cost: Cost of this request (default: 1)
            
        Returns:
            True if request is allowed, False otherwise
        """
        key = key or "global"
        
        async with self._lock:
            now = time.time()
            
            # Initialize request times if not exists
            if key not in self.request_times:
                self.request_times[key] = []
            
            # Remove old requests outside the window
            self.request_times[key] = [t for t in self.request_times[key] 
                                     if now - t <= self.window_size]
            
            # Check if adding this request would exceed the rate
            current_count = len(self.request_times[key])
            max_count = self.rate * self.window_size
            
            if current_count + cost <= max_count:
                # Request allowed - add timestamps for each unit of cost
                for _ in range(cost):
                    self.request_times[key].append(now)
                return True
            else:
                return False
    
    async def wait(self, key: Optional[str] = None, cost: int = 1) -> float:
        """
        Wait until a request is allowed to proceed.
        
        Args:
            key: Optional key for partitioned rate limiting
            cost: Cost of this request (default: 1)
            
        Returns:
            Wait time in seconds
        """
        key = key or "global"
        start_time = time.time()
        
        while True:
            async with self._lock:
                now = time.time()
                
                # Initialize request times if not exists
                if key not in self.request_times:
                    self.request_times[key] = []
                
                # Remove old requests outside the window
                self.request_times[key] = [t for t in self.request_times[key] 
                                         if now - t <= self.window_size]
                
                # Check if adding this request would exceed the rate
                current_count = len(self.request_times[key])
                max_count = self.rate * self.window_size
                
                if current_count + cost <= max_count:
                    # Request allowed - add timestamps for each unit of cost
                    for _ in range(cost):
                        self.request_times[key].append(now)
                    wait_time = now - start_time
                    return wait_time
                else:
                    # Calculate time until enough tokens will be available
                    oldest_request = min(self.request_times[key])
                    wait_time = oldest_request + self.window_size - now
                    wait_time = max(0.01, wait_time)  # At least 10ms
            
            # Wait a bit before checking again
            await asyncio.sleep(min(wait_time, 0.1))
    
    def update_rate(self, rate: float) -> None:
        """
        Update the rate limit.
        
        Args:
            rate: New rate limit (requests per second)
        """
        self.rate = max(0.1, rate)  # Ensure rate is at least 0.1 req/sec

class TokenBucketRateLimiter(BaseRateLimiter):
    """
    Token bucket rate limiter that allows for bursts of traffic.
    
    This implementation uses a token bucket algorithm where tokens are added
    at a fixed rate up to a maximum capacity. Each request consumes one or more tokens.
    """
    
    def __init__(self, 
                rate: float = 10.0, 
                bucket_capacity: int = 20,
                initial_tokens: Optional[int] = None):
        """
        Initialize the token bucket rate limiter.
        
        Args:
            rate: Token replenishment rate (tokens per second)
            bucket_capacity: Maximum number of tokens in the bucket
            initial_tokens: Initial number of tokens (default: full bucket)
        """
        self.rate = rate
        self.bucket_capacity = bucket_capacity
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        
        self.initial_tokens = initial_tokens
        if self.initial_tokens is None:
            self.initial_tokens = bucket_capacity
    
    def _get_bucket(self, key: str) -> Dict[str, Any]:
        """
        Get or create a bucket for a key.
        
        Args:
            key: Bucket key
            
        Returns:
            Bucket data
        """
        if key not in self.buckets:
            self.buckets[key] = {
                "tokens": self.initial_tokens,
                "last_updated": time.time()
            }
        return self.buckets[key]
    
    def _update_tokens(self, bucket: Dict[str, Any]) -> None:
        """
        Update the tokens in a bucket based on elapsed time.
        
        Args:
            bucket: Bucket data to update
        """
        now = time.time()
        elapsed = now - bucket["last_updated"]
        
        # Calculate new tokens based on rate and elapsed time
        new_tokens = bucket["tokens"] + (elapsed * self.rate)
        
        # Cap at bucket capacity
        bucket["tokens"] = min(new_tokens, self.bucket_capacity)
        bucket["last_updated"] = now
    
    async def acquire(self, key: Optional[str] = None, cost: int = 1) -> bool:
        """
        Check if a request is allowed without waiting.
        
        Args:
            key: Optional key for partitioned rate limiting
            cost: Number of tokens to consume (default: 1)
            
        Returns:
            True if request is allowed, False otherwise
        """
        key = key or "global"
        
        async with self._lock:
            bucket = self._get_bucket(key)
            self._update_tokens(bucket)
            
            if bucket["tokens"] >= cost:
                bucket["tokens"] -= cost
                return True
            else:
                return False
    
    async def wait(self, key: Optional[str] = None, cost: int = 1) -> float:
        """
        Wait until enough tokens are available.
        
        Args:
            key: Optional key for partitioned rate limiting
            cost: Number of tokens to consume (default: 1)
            
        Returns:
            Wait time in seconds
        """
        key = key or "global"
        start_time = time.time()
        
        while True:
            async with self._lock:
                bucket = self._get_bucket(key)
                self._update_tokens(bucket)
                
                if bucket["tokens"] >= cost:
                    bucket["tokens"] -= cost
                    wait_time = time.time() - start_time
                    return wait_time
                else:
                    # Calculate time until enough tokens will be available
                    missing_tokens = cost - bucket["tokens"]
                    wait_time = missing_tokens / self.rate
                    wait_time = max(0.01, wait_time)  # At least 10ms
            
            # Wait a bit before checking again
            await asyncio.sleep(min(wait_time, 0.1))
    
    def update_rate(self, rate: float) -> None:
        """
        Update the token replenishment rate.
        
        Args:
            rate: New token replenishment rate (tokens per second)
        """
        self.rate = max(0.1, rate)  # Ensure rate is at least 0.1 tokens/sec

class AdaptiveRateLimiter(BaseRateLimiter):
    """
    Adaptive rate limiter that adjusts based on service response.
    
    This implementation uses a token bucket algorithm but adjusts the rate
    based on response times, error rates, and server hints (like 429 status codes).
    """
    
    def __init__(self, 
                initial_rate: float = 10.0, 
                min_rate: float = 1.0,
                max_rate: float = 50.0,
                bucket_capacity: int = 20,
                window_size: int = 20,
                target_success_rate: float = 0.95,
                adjustment_factor: float = 0.1):
        """
        Initialize the adaptive rate limiter.
        
        Args:
            initial_rate: Initial token rate (tokens per second)
            min_rate: Minimum token rate
            max_rate: Maximum token rate
            bucket_capacity: Maximum tokens in the bucket
            window_size: Number of requests to consider for adaptation
            target_success_rate: Target success rate (0-1)
            adjustment_factor: How quickly to adjust the rate (0-1)
        """
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.bucket_capacity = bucket_capacity
        self.window_size = window_size
        self.target_success_rate = target_success_rate
        self.adjustment_factor = adjustment_factor
        
        # Use token bucket as base limiter
        self.token_bucket = TokenBucketRateLimiter(
            rate=initial_rate,
            bucket_capacity=bucket_capacity
        )
        
        # Track response data for adaptation
        self.response_history: Dict[str, List[Dict[str, Any]]] = {}
        self.current_rates: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
        # Backoff state
        self.backoff_until: Dict[str, float] = {}
    
    async def acquire(self, key: Optional[str] = None, cost: int = 1) -> bool:
        """
        Check if a request is allowed without waiting.
        
        Args:
            key: Optional key for partitioned rate limiting
            cost: Number of tokens to consume (default: 1)
            
        Returns:
            True if request is allowed, False otherwise
        """
        key = key or "global"
        
        async with self._lock:
            # Check if we're in backoff period
            now = time.time()
            if key in self.backoff_until and now < self.backoff_until[key]:
                return False
            
            # Otherwise defer to token bucket
            return await self.token_bucket.acquire(key, cost)
    
    async def wait(self, key: Optional[str] = None, cost: int = 1) -> float:
        """
        Wait until a request is allowed to proceed.
        
        Args:
            key: Optional key for partitioned rate limiting
            cost: Number of tokens to consume (default: 1)
            
        Returns:
            Wait time in seconds
        """
        key = key or "global"
        start_time = time.time()
        
        # Check if we're in backoff period
        now = time.time()
        if key in self.backoff_until and now < self.backoff_until[key]:
            wait_time = self.backoff_until[key] - now
            wait_time = max(0.01, wait_time)  # At least 10ms
            await asyncio.sleep(wait_time)
        
        # Wait for token bucket rate limiter
        await self.token_bucket.wait(key, cost)
        return time.time() - start_time
    
    def update_rate(self, rate: float, key: Optional[str] = None) -> None:
        """
        Update the rate limit for a specific key.
        
        Args:
            rate: New rate limit (requests per second)
            key: Optional key for partitioned rate limiting
        """
        key = key or "global"
        rate = max(self.min_rate, min(self.max_rate, rate))
        
        self.current_rates[key] = rate
        
        # Also update the underlying token bucket
        if key == "global":
            self.token_bucket.update_rate(rate)
    
    def record_response(self, 
                      success: bool, 
                      response_time: float, 
                      status_code: Optional[int] = None,
                      retry_after: Optional[int] = None,
                      key: Optional[str] = None) -> None:
        """
        Record a response to adapt the rate limit.
        
        Args:
            success: Whether the request was successful
            response_time: Response time in seconds
            status_code: Optional HTTP status code
            retry_after: Optional retry-after value from response headers
            key: Optional key for partitioned rate limiting
        """
        key = key or "global"
        
        # Initialize history for this key if needed
        if key not in self.response_history:
            self.response_history[key] = []
            self.current_rates[key] = self.initial_rate
        
        # Add response to history
        self.response_history[key].append({
            "success": success,
            "response_time": response_time,
            "status_code": status_code,
            "timestamp": time.time()
        })
        
        # Trim history to window size
        self.response_history[key] = self.response_history[key][-self.window_size:]
        
        # Handle rate limit responses (429) with retry-after
        if status_code == 429 and retry_after is not None:
            # Apply immediate backoff based on retry-after
            self.backoff_until[key] = time.time() + retry_after
            
            # Also reduce current rate
            current_rate = self.current_rates.get(key, self.initial_rate)
            new_rate = max(self.min_rate, current_rate * 0.5)  # Reduce by 50%
            self.update_rate(new_rate, key)
            
            logger.info(f"Rate limit hit for {key}, backing off for {retry_after}s, reducing rate to {new_rate}")
            return
        
        # Adapt rate based on success rate and response times
        self._adapt_rate(key)
    
    def _adapt_rate(self, key: str) -> None:
        """
        Adapt the rate limit based on recent history.
        
        Args:
            key: Key for partitioned rate limiting
        """
        if len(self.response_history[key]) < 5:
            # Not enough data to adapt yet
            return
        
        # Calculate success rate
        successes = sum(1 for r in self.response_history[key] if r["success"])
        success_rate = successes / len(self.response_history[key])
        
        # Calculate average response time
        avg_time = sum(r["response_time"] for r in self.response_history[key]) / len(self.response_history[key])
        
        # Get current rate
        current_rate = self.current_rates.get(key, self.initial_rate)
        
        # Adjust rate based on success rate
        if success_rate < self.target_success_rate:
            # Reduce rate
            rate_change = -self.adjustment_factor * current_rate * (self.target_success_rate - success_rate)
        else:
            # Increase rate slowly
            rate_change = self.adjustment_factor * current_rate * 0.05
        
        # Apply adjustment with bounds
        new_rate = max(self.min_rate, min(self.max_rate, current_rate + rate_change))
        
        # If response time is increasing, be more conservative
        recent_times = [r["response_time"] for r in self.response_history[key][-5:]]
        if len(recent_times) >= 5 and all(recent_times[i] > recent_times[i-1] for i in range(1, 5)):
            # Response times are consistently increasing
            new_rate = min(new_rate, current_rate)  # Don't increase rate
        
        # Apply new rate
        if abs(new_rate - current_rate) / current_rate > 0.02:  # Only log if change is > 2%
            logger.info(f"Adapting rate for {key}: {current_rate:.2f} → {new_rate:.2f} req/s " +
                       f"(success rate: {success_rate:.2f}, avg time: {avg_time:.2f}s)")
        
        self.update_rate(new_rate, key)

class ThrottledClientMixin:
    """
    Mixin class to add throttling capabilities to any HTTP client.
    
    Usage:
        class ThrottledHttpClient(ThrottledClientMixin, BaseHttpClient):
            pass
    """
    
    def __init__(self, *args, **kwargs):
        # Extract rate limiting args
        self.rate_limit = kwargs.pop('rate_limit', 10.0)
        self.rate_limit_strategy = kwargs.pop('rate_limit_strategy', 'token_bucket')
        self.max_tokens = kwargs.pop('max_tokens', self.rate_limit * 2)
        self.adaptive = kwargs.pop('adaptive', True)
        
        # Initialize parent class
        super().__init__(*args, **kwargs)
        
        # Create rate limiter based on strategy
        if self.rate_limit_strategy == 'token_bucket':
            if self.adaptive:
                self.rate_limiter = AdaptiveRateLimiter(
                    initial_rate=self.rate_limit,
                    bucket_capacity=self.max_tokens
                )
            else:
                self.rate_limiter = TokenBucketRateLimiter(
                    rate=self.rate_limit,
                    bucket_capacity=self.max_tokens
                )
        else:  # simple
            self.rate_limiter = SimpleRateLimiter(
                rate=self.rate_limit
            )
    
    async def throttled_request(self, request_func, *args, **kwargs):
        """
        Execute a request with throttling.
        
        Args:
            request_func: Function to execute
            *args: Arguments for the request function
            **kwargs: Keyword arguments for the request function
            
        Returns:
            Response from the request function
        """
        # Wait for rate limiter
        await self.rate_limiter.wait()
        
        # Track timing and execution
        start_time = time.time()
        success = False
        status_code = None
        retry_after = None
        
        try:
            # Execute request
            response = await request_func(*args, **kwargs)
            
            # Process response
            success = 200 <= response.status_code < 400
            status_code = response.status_code
            
            # Check for retry-after header
            if hasattr(response, 'headers') and 'retry-after' in response.headers:
                try:
                    retry_after = int(response.headers['retry-after'])
                except (ValueError, TypeError):
                    pass
            
            return response
        except Exception as e:
            # Handle exceptions
            success = False
            
            # Try to extract status code if available
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            
            # Try to extract retry-after if available
            if hasattr(e, 'headers') and 'retry-after' in e.headers:
                try:
                    retry_after = int(e.headers['retry-after'])
                except (ValueError, TypeError):
                    pass
            
            raise
        finally:
            # Record response for adaptation
            response_time = time.time() - start_time
            if hasattr(self.rate_limiter, 'record_response'):
                self.rate_limiter.record_response(
                    success=success,
                    response_time=response_time,
                    status_code=status_code,
                    retry_after=retry_after
                )

# Export the classes and mixins
__all__ = [
    'BaseRateLimiter',
    'SimpleRateLimiter',
    'TokenBucketRateLimiter',
    'AdaptiveRateLimiter',
    'ThrottledClientMixin'
]