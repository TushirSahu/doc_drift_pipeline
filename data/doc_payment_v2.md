# Payment API Documentation

## Overview
The Payment API provides endpoints for processing payments, managing transactions, and retrieving payment history.

## Base URL Information
```
https://api.payment.example.com/v1
```

## Authentication
All requests require Bearer token authentication in the Authorization header.

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### Process Payment
**POST** `/payments/process`

Process a new payment transaction.

**Request Body:**
```json
{
  "amount": 99.99,
  "currency": "USD",
  "payment_method": "credit_card",
  "card_token": "tok_1234567890",
  "customer_id": "cust_abc123",
  "description": "Order #12345"
}
```

**Response:**
```json
{
  "transaction_id": "txn_1234567890",
  "status": "success",
  "amount": 99.99,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Get Transaction
**GET** `/payments/{transaction_id}`

Retrieve transaction details.

**Response:**
```json
{
  "transaction_id": "txn_1234567890",
  "amount": 99.99,
  "status": "completed",
  "payment_method": "credit_card",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### List Payments
**GET** `/payments?customer_id={customer_id}&limit=10`

List payments for a customer.

**Response:**
```json
{
  "data": [
    {
      "transaction_id": "txn_1234567890",
      "amount": 99.99,
      "status": "completed"
    }
  ],
  "total": 1
}
```

## Error Codes
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `500` - Internal Server Error
