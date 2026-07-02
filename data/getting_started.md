# Getting Started

## Overview
The Example API Platform lets you process payments, manage authentication, and
receive event notifications. This guide covers the basics you need to make your
first request.

## Base URL
All API requests use this base URL:

```
https://api.payment.example.com/v1
```

## Authentication
Two mechanisms are available:

- **Bearer API key** — pass your key in the `Authorization` header
  (`Authorization: Bearer YOUR_API_KEY`). Best for server-to-server calls.
- **OAuth2 + JWT** — issued by Auth Service v2.0. Access tokens expire after 15
  minutes; refresh them at `/api/v2/auth/refresh`. Best for user-facing apps.

Never expose your API key in client-side code.

## Making your first request
Send a test request to confirm your credentials work:

```
GET /v1/ping
Authorization: Bearer YOUR_API_KEY
```

A successful response returns `{"status": "ok"}` with HTTP 200.

## Environments
- **Sandbox** — `https://sandbox.api.payment.example.com/v1`, for testing with
  fake card numbers. No real money moves.
- **Production** — `https://api.payment.example.com/v1`, for live transactions.

Use sandbox keys (prefixed `sk_test_`) while developing and production keys
(prefixed `sk_live_`) when you go live.

## Versioning
The API is versioned in the URL path (`/v1`). Breaking changes ship under a new
version; older versions are supported for at least 12 months after deprecation.
