"""Tests for authentication endpoints."""

import pytest


class TestAuthEndpoints:
    """Test authentication endpoints."""

    def test_register_user(self, api_client, sample_user_data):
        """Test user registration endpoint."""
        # TODO: Implement test
        # response = api_client.post("/auth/register", json=sample_user_data)
        # assert response.status_code == 201
        # assert "access_token" in response.json()
        pass

    def test_login_user(self, api_client, sample_user_data):
        """Test user login endpoint."""
        # TODO: Implement test
        # First register
        # api_client.post("/auth/register", json=sample_user_data)
        # Then login
        # response = api_client.post("/auth/login", json={"email": sample_user_data["email"]})
        # assert response.status_code == 200
        # assert "access_token" in response.json()
        pass

    def test_login_nonexistent_user(self, api_client):
        """Test login with non-existent user."""
        # TODO: Implement test
        # response = api_client.post("/auth/login", json={"email": "nonexistent@example.com"})
        # assert response.status_code == 401
        pass

    def test_register_duplicate_email(self, api_client, sample_user_data):
        """Test registering with duplicate email."""
        # TODO: Implement test
        # api_client.post("/auth/register", json=sample_user_data)
        # response = api_client.post("/auth/register", json=sample_user_data)
        # assert response.status_code == 400
        pass
