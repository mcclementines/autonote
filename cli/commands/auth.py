"""Authentication command handlers."""

import httpx
from ..config import API_URL, save_token, delete_token, delete_session


def register_user():
    """Handle user registration and auto-login."""
    print("\n=== User Registration ===")
    email = input("Email: ").strip()
    name = input("Name: ").strip()

    if not email or not name:
        print("Error: Email and name are required.\n")
        return

    try:
        response = httpx.post(
            f"{API_URL}/auth/register",
            json={"email": email, "name": name},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        # Save token
        save_token(data["access_token"])

        user = data["user"]
        print(f"\n✓ Registration successful! You are now logged in.")
        print(f"  User ID: {user['id']}")
        print(f"  Email: {user['email']}")
        print(f"  Name: {user['name']}\n")
    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            print(f"Error: {e.response.json().get('detail', 'Registration failed')}\n")
        else:
            print(f"Error: Registration failed: {e}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def login_user():
    """Handle user login."""
    print("\n=== User Login ===")
    email = input("Email: ").strip()

    if not email:
        print("Error: Email is required.\n")
        return

    try:
        response = httpx.post(
            f"{API_URL}/auth/login",
            json={"email": email},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

        # Save token
        save_token(data["access_token"])

        user = data["user"]
        print(f"\n✓ Login successful!")
        print(f"  Welcome back, {user['name']}!\n")
    except httpx.ConnectError:
        print("Error: Could not connect to API server.")
        print("Please start the server with: python -m api.server\n")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Error: Invalid credentials. User not found.\n")
        elif e.response.status_code == 403:
            print("Error: Your account has been disabled.\n")
        else:
            print(f"Error: Login failed: {e}\n")
    except httpx.HTTPError as e:
        print(f"Error: API request failed: {e}\n")


def logout_user():
    """Handle user logout."""
    delete_token()
    delete_session()
    print("\n✓ Logged out successfully.\n")
