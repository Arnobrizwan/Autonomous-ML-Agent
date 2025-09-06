#!/bin/bash

# 🚀 One-Command Demo Starter
# This starts everything you need for a demo

echo "🚀 Starting Autonomous ML Agent Demo..."
echo "====================================="

# Start Docker container
echo "Starting Docker container..."
docker run -d --name aml-demo -p 8000:8000 aml-agent:latest

# Wait for container to start
echo "Waiting for service to start..."
sleep 5

# Check health
echo "Checking health..."
if curl -s http://localhost:8000/healthz > /dev/null; then
    echo "✅ Service is running!"
    echo ""
    echo "🌐 Demo URLs:"
    echo "  • Web Interface: http://localhost:8000"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • Health Check: http://localhost:8000/healthz"
    echo ""
    echo "🎯 Ready for your demo!"
    echo ""
    echo "💡 To stop: docker stop aml-demo && docker rm aml-demo"
    echo "💡 To run full demo: ./demo.sh"
else
    echo "❌ Service failed to start"
    echo "Check logs: docker logs aml-demo"
fi
