# Webhooks

## Overview
Webhooks let the platform notify your server when events happen (for example, a
payment succeeds) instead of you polling the API.

## Registering an endpoint
Create a webhook by sending your HTTPS URL:

```
POST /webhooks
{
  "url": "https://yourapp.com/hooks/payments",
  "events": ["payment.succeeded", "payment.failed", "refund.created"]
}
```

The endpoint must be publicly reachable and respond with HTTP 200 within 5
seconds, or the delivery is treated as failed.

## Event types
- `payment.succeeded` — a payment completed successfully.
- `payment.failed` — a payment attempt was declined.
- `refund.created` — a refund was issued.
- `refund.completed` — a refund settled to the customer.

## Payload format
Each delivery is a JSON body:

```json
{
  "id": "evt_123",
  "type": "payment.succeeded",
  "created": "2024-01-15T10:30:00Z",
  "data": { "transaction_id": "txn_123", "amount": 99.99, "currency": "USD" }
}
```

## Verifying signatures
Every request includes an `X-Signature` header — an HMAC-SHA256 of the raw body
using your webhook signing secret. Recompute it and compare; reject the request
if it doesn't match. This prevents spoofed events.

## Retries
Failed deliveries are retried with exponential backoff for up to 24 hours. After
that the event is marked undelivered. You can replay events from the dashboard.
