#!/bin/bash
# ==============================================================================
# SIH Dead Reckoning Backend - VPS Automated Deployment Script
# Supports: Ubuntu 20.04/22.04/24.04, Debian 11/12, Amazon Linux 2023
# ==============================================================================

set -e

echo "========================================================="
echo " Starting SIH Dead Reckoning Backend Deployment..."
echo "========================================================="

# 1. Update system packages non-interactively
echo "[1/6] Updating package repositories..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y || sudo yum update -y

# 2. Check and install Docker & Docker Compose if missing
if ! command -v docker &> /dev/null; then
    echo "[2/6] Docker not found. Installing Docker Engine..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm -f get-docker.sh
    sudo usermod -aG docker $USER || true
    echo "Docker installed successfully."
else
    echo "[2/6] Docker is already installed: $(docker --version)"
fi

# 3. Verify Docker Compose plugin
echo "[3/6] Verifying Docker Compose..."
if ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose plugin..."
    sudo apt-get install -y docker-compose-plugin || sudo yum install -y docker-compose-plugin
fi

# 4. Configure Firewall if UFW is active
if command -v ufw &> /dev/null && sudo ufw status | grep -q "Status: active"; then
    echo "Configuring UFW firewall for ports 80, 443, 8000..."
    sudo ufw allow 80/tcp || true
    sudo ufw allow 443/tcp || true
    sudo ufw allow 8000/tcp || true
fi

# 5. Prepare Environment Variables
if [ ! -f .env ]; then
    echo "[4/6] Creating .env from .env.example..."
    cp .env.example .env
    # Generate secure random JWT secret
    SECRET=$(openssl rand -hex 24 2>/dev/null || date +%s%N)
    sed -i "s/change-me-generate-with-openssl-rand-hex-32/dr_jwt_$SECRET/g" .env
else
    echo "[4/6] Using existing .env configuration."
fi

# 6. Build and Launch Containers
echo "[5/6] Building and launching multi-container stack (FastAPI + Redis + TimescaleDB)..."
sudo docker compose down --remove-orphans || true
sudo docker compose up -d --build

# 7. Verification and Healthcheck
echo "[6/6] Waiting for services to stabilize (10s)..."
sleep 10

echo ""
echo "========================================================="
echo " Deployment Complete! Checking container statuses:"
echo "========================================================="
sudo docker compose ps

PUBLIC_IP=$(curl -s -m 5 https://api.ipify.org || curl -s -m 5 ifconfig.me || echo "YOUR_VPS_IP")

echo ""
echo "Testing Backend Health Endpoint:"
curl -s http://localhost:8000/health || echo "Endpoint starting up..."
echo ""
echo "========================================================="
echo " SIH 2026 Dead Reckoning Backend is now LIVE!"
echo " Web Dashboard: http://${PUBLIC_IP}:8000/"
echo " API Docs:      http://${PUBLIC_IP}:8000/docs"
echo " Health Status: http://${PUBLIC_IP}:8000/health"
echo " WebSocket URL: ws://${PUBLIC_IP}:8000/ws/track/{device_id}?token=<jwt>"
echo "========================================================="
