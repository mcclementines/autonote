"""Tests for authentication endpoints."""

import pytest


class TestAuthEndpoints:
    """Test authentication endpoints."""

    def test_register_user(self, api_client, sample_user_data):
        """Test user registration endpoint."""
        response = api_client.post("/auth/register", json=sample_user_data)

        assert response.status_code == 201
        data = response.json()

        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["email"] == sample_user_data["email"]
        assert data["user"]["name"] == sample_user_data["name"]
        assert data["user"]["status"] == "active"

    def test_login_user(self, api_client, sample_user_data):
        """Test user login endpoint."""
        # First register
        api_client.post("/auth/register", json=sample_user_data)

        # Then login
        response = api_client.post("/auth/login", json={"email": sample_user_data["email"]})

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["email"] == sample_user_data["email"]

    def test_login_nonexistent_user(self, api_client):
        """Test login with non-existent user."""
        response = api_client.post("/auth/login", json={"email": "nonexistent@example.com"})

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_register_duplicate_email(self, api_client, sample_user_data):
        """Test registering with duplicate email."""
        # Register first time
        api_client.post("/auth/register", json=sample_user_data)

        # Try to register again with same email
        response = api_client.post("/auth/register", json=sample_user_data)

        assert response.status_code == 400
        assert response.json()["detail"] == "Email already registered"

    def test_register_invalid_email(self, api_client):
        """Test registration with invalid email format."""
        response = api_client.post("/auth/register", json={
            "email": "not-an-email",
            "name": "Test User"
        })

        assert response.status_code == 422  # Validation error
