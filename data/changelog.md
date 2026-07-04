# Changelog

## v3.0 (current)
- **Auth:** OAuth2 + JWT is now the default. Access tokens expire after 15
  minutes; refresh at `/api/v2/auth/refresh`.
- **Payments:** added the Refunds API (`POST /refunds`), including partial refunds.
- **Webhooks:** new `refund.created` and `refund.completed` events.
- **Users:** added role-based access control (`admin`, `member`, `viewer`).
- **Breaking:** the legacy `/v1/charge` endpoint is removed — use
  `POST /payments/process`.

## v2.0
- Introduced Bearer API-key authentication.
- Added the Payment API (`/payments/process`, `/payments/{id}`).
- Session cookies deprecated in favor of tokens.

## v1.0
- Initial release: basic payment processing with session-cookie auth.

## Migration notes: v2 → v3
- Replace any calls to `/v1/charge` with `POST /payments/process`.
- Switch from long-lived API keys to OAuth2 tokens where user context is needed;
  server-to-server calls may continue using Bearer keys.
- Refunds are issued in the original payment currency; cross-currency refunds are
  not supported.
