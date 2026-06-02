
    # Auth Service v2.0
    The new Auth Service v2.0 uses OAuth2 and JWT tokens instead of session cookies. 
    Tokens expire after 15 minutes by default. 
    To refresh a token, you must hit the `/api/v2/auth/refresh` endpoint.
    Admin users have an extended maximum session time of 12 hours.
    