# Refunds API

## Overview
The Refunds API reverses a payment, fully or partially, back to the customer's
original payment method.

## Create a refund
**POST** `/refunds`

```json
{
  "transaction_id": "txn_1234567890",
  "amount": 49.99,
  "reason": "customer_request"
}
```

- Omit `amount` to refund the full transaction.
- Include a smaller `amount` for a partial refund.
- Refunds are issued in the **same currency** as the original payment (for
  example, a payment made in EUR is refunded in EUR). Cross-currency refunds are
  not supported.

**Response:**

```json
{
  "refund_id": "rfnd_987",
  "transaction_id": "txn_1234567890",
  "amount": 49.99,
  "currency": "EUR",
  "status": "pending"
}
```

## Refund status
- `pending` — submitted, not yet settled.
- `completed` — funds returned to the customer.
- `failed` — the refund could not be processed.

Settlement typically takes 5–10 business days depending on the customer's bank.

## Rules and limits
- You can only refund a payment with status `completed`.
- Total refunds cannot exceed the original payment amount.
- Refunds can be issued up to 180 days after the original payment.
