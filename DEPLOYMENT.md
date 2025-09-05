# Production Deployment Guide

This guide covers deploying the Autonomous ML Agent to production environments.

## Quick Start

### Local Development
```bash
# Setup environment
make setup

# Run the pipeline
make run

# Start API service
make serve
```

### Docker Deployment
```bash
# Build and run with Docker
make docker-build
make docker-run

# Or use docker-compose for full stack
make docker-compose-up
```

## Production Deployment

### 1. Docker Hub Deployment

1. **Set up Docker Hub credentials:**
   ```bash
   export DOCKER_USERNAME=your-username
   docker login
   ```

2. **Build and push production image:**
   ```bash
   make docker-build-prod
   make docker-push
   ```

3. **Deploy to production server:**
   ```bash
   docker pull your-username/aml-agent:latest
   docker run -d --name aml-agent \
     -p 8000:8000 \
     -v /path/to/data:/app/data \
     -v /path/to/artifacts:/app/artifacts \
     your-username/aml-agent:latest
   ```

### 2. Kubernetes Deployment

Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aml-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aml-agent
  template:
    metadata:
      labels:
        app: aml-agent
    spec:
      containers:
      - name: aml-agent
        image: your-username/aml-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATA_PATH
          value: "/app/data"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: artifacts-volume
          mountPath: /app/artifacts
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: aml-agent-data
      - name: artifacts-volume
        persistentVolumeClaim:
          claimName: aml-agent-artifacts
---
apiVersion: v1
kind: Service
metadata:
  name: aml-agent-service
spec:
  selector:
    app: aml-agent
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s/deployment.yaml
```

### 3. AWS ECS Deployment

1. **Create ECR repository:**
   ```bash
   aws ecr create-repository --repository-name aml-agent
   ```

2. **Build and push to ECR:**
   ```bash
   aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
   docker tag aml-agent:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/aml-agent:latest
   docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/aml-agent:latest
   ```

3. **Create ECS task definition and service**

### 4. Google Cloud Run

1. **Build and push to GCR:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/aml-agent
   ```

2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy aml-agent \
     --image gcr.io/PROJECT-ID/aml-agent \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## Monitoring and Observability

### Health Checks

The service provides health check endpoints:
- `GET /healthz` - Basic health check
- `GET /health` - Detailed health status
- `GET /metrics` - Prometheus metrics

### Logging

Logs are structured JSON and include:
- Request/response details
- Model performance metrics
- Error traces
- Training progress

### Metrics

Key metrics to monitor:
- Request rate and latency
- Model prediction accuracy
- Training job success rate
- Resource utilization

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_PATH` | Path to data directory | `/app/data` |
| `ARTIFACTS_PATH` | Path to artifacts directory | `/app/artifacts` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_HOST` | API host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |

### Configuration Files

- `configs/default.yaml` - Default configuration
- `configs/production.yaml` - Production overrides
- `.env` - Environment-specific variables

## Security

### API Security
- Input validation with Pydantic
- Rate limiting (configure in nginx)
- CORS configuration
- Authentication (add as needed)

### Data Security
- Encrypt data at rest
- Secure data transmission (HTTPS)
- Access control for data directories
- Regular security updates

## Scaling

### Horizontal Scaling
- Use load balancer (nginx, ALB, etc.)
- Multiple container instances
- Stateless design (no local state)

### Vertical Scaling
- Increase container resources
- Optimize model inference
- Use GPU acceleration for training

## Backup and Recovery

### Data Backup
```bash
# Backup data directory
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/

# Backup artifacts
tar -czf artifacts-backup-$(date +%Y%m%d).tar.gz artifacts/
```

### Model Backup
- Models are automatically saved to artifacts directory
- Version control with git tags
- Export to external storage (S3, GCS, etc.)

## Troubleshooting

### Common Issues

1. **Container won't start:**
   ```bash
   docker logs aml-agent
   ```

2. **API not responding:**
   ```bash
   curl http://localhost:8000/healthz
   ```

3. **Model loading errors:**
   - Check artifacts directory permissions
   - Verify model files exist
   - Check model compatibility

### Debug Mode

Run with debug logging:
```bash
docker run -e LOG_LEVEL=DEBUG aml-agent:latest
```

## Performance Tuning

### Model Optimization
- Use ONNX for faster inference
- Quantize models for smaller size
- Batch predictions when possible

### Infrastructure Optimization
- Use SSD storage for data
- Allocate sufficient memory
- Use multi-core CPUs
- Consider GPU acceleration

## Maintenance

### Regular Tasks
- Monitor disk space usage
- Update dependencies
- Backup data and models
- Review logs for errors

### Updates
- Test updates in staging
- Use blue-green deployment
- Rollback plan ready
- Monitor after deployment
