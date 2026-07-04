# Security Best Practices

## API keys
- Keep API keys secret. Never commit them to source control or expose them in
  client-side code.
- Use **test keys** (`sk_test_`) in development and **live keys** (`sk_live_`) in
  production.
- Rotate keys every 90 days, and immediately if a key may have leaked.

## Scopes
Keys can be scoped to limit access:
- `read` — read-only endpoints.
- `write` — create and update resources.
- `admin` — user and billing management.

Grant the narrowest scope a key needs. A public-facing service should use a
`read`-only key.

## IP allowlisting
Restrict a key to specific IP ranges from the dashboard. Requests from other
addresses are rejected with `403 Forbidden`. Recommended for server-to-server
integrations.

## Webhook verification
Always verify the `X-Signature` header on incoming webhooks (HMAC-SHA256 of the
raw body with your signing secret). Reject requests whose signature doesn't match
to prevent spoofed events.

## Rate limits & abuse
Requests are rate limited per key. Handle `429 Too Many Requests` with the
`Retry-After` header and exponential backoff. Repeated abuse can lead to a
temporary key suspension.

## Data handling
- All traffic uses TLS 1.2+; plain HTTP requests are refused.
- Card data is tokenized — raw card numbers are never stored.
- Personal data can be exported or deleted on request under your data-retention
  settings.
