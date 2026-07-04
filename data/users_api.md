# Users API

## Overview
The Users API manages user accounts, roles, and their access to the platform.

## Create a user
**POST** `/users`

```json
{
  "email": "jane@example.com",
  "name": "Jane Doe",
  "role": "member"
}
```

Roles are one of `admin`, `member`, or `viewer`. New users default to `member`.
The response returns a `user_id` and a `created_at` timestamp.

## Get a user
**GET** `/users/{user_id}`

Returns the user's profile, role, and status (`active`, `suspended`, or `invited`).

## List users
**GET** `/users?role=admin&limit=20`

Returns a paginated list. Filter by `role` or `status`. Maximum `limit` is 100.

## Update a role
**PATCH** `/users/{user_id}`

```json
{ "role": "admin" }
```

Only an `admin` can change roles. Attempting this as a `member` returns
`403 Forbidden`.

## Deactivate a user
**DELETE** `/users/{user_id}`

Soft-deletes the account (sets status to `suspended`); it can be reactivated
within 30 days, after which the data is permanently removed.

## Roles & permissions
- `admin` — full access, can manage users and billing.
- `member` — can use the API and view their own data.
- `viewer` — read-only access to shared resources.
