# Rate Limits & Quotas

## Overview
The API enforces rate limits to keep the platform stable. Limits are applied
per API key.

## Limits by plan
- **Free** — 60 requests per minute, 10,000 requests per month.
- **Pro** — 600 requests per minute, 1,000,000 requests per month.
- **Enterprise** — custom limits, negotiated per contract.

## Rate limit headers
Every response includes your current usage:

```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 597
X-RateLimit-Reset: 1700000000
```

`X-RateLimit-Reset` is a Unix timestamp for when the window resets.

## Handling 429 responses
When you exceed the limit, the API returns HTTP `429 Too Many Requests`. The
response includes a `Retry-After` header (in seconds). Wait that long before
retrying, and use exponential backoff for repeated failures.

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Retry after 30 seconds."
}
```

## Best practices
- Cache responses where possible to reduce request volume.
- Batch operations instead of sending many small requests.
- Spread scheduled jobs out instead of firing them all at once.
