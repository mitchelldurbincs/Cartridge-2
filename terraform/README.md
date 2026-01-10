# Cartridge2 Terraform Infrastructure

This directory contains Terraform configurations for deploying Cartridge2 to Google Cloud Platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Google Cloud                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    VPC Network                             │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              GKE Autopilot Cluster                   │  │  │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐  │  │  │
│  │  │  │ Actors  │ │ Trainer │ │   Web   │ │ Frontend  │  │  │  │
│  │  │  │ (2-16)  │ │  (1)    │ │  (2)    │ │   (2)     │  │  │  │
│  │  │  └────┬────┘ └────┬────┘ └────┬────┘ └─────┬─────┘  │  │  │
│  │  └───────┼───────────┼───────────┼────────────┼────────┘  │  │
│  │          │           │           │            │           │  │
│  │  ┌───────▼───────────▼───────────▼────────────▼───────┐  │  │
│  │  │              Cloud Load Balancer                    │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │  Cloud SQL    │  │    GCS       │  │  Artifact Registry │   │
│  │  (PostgreSQL) │  │  (Models)    │  │  (Images)          │   │
│  └───────────────┘  └──────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. [Terraform](https://www.terraform.io/downloads) >= 1.5.0
2. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
3. A GCP project with billing enabled

## Quick Start

### 1. Authenticate with GCP

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required APIs

```bash
gcloud services enable \
  container.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  servicenetworking.googleapis.com \
  compute.googleapis.com
```

### 3. Initialize and Apply (Dev)

```bash
cd terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID

terraform init
terraform plan
terraform apply
```

### 4. Deploy Workloads

After infrastructure is provisioned:

```bash
# Get cluster credentials
gcloud container clusters get-credentials cartridge-dev --region us-central1

# Apply Kubernetes manifests
kubectl apply -k ../../k8s/overlays/dev
```

## Directory Structure

```
terraform/
├── environments/
│   ├── dev/                    # Development environment
│   │   ├── main.tf             # Module composition
│   │   ├── variables.tf        # Variable definitions
│   │   ├── outputs.tf          # Output values
│   │   └── terraform.tfvars    # Environment-specific values
│   └── prod/                   # Production environment
│       └── ...
└── modules/
    ├── networking/             # VPC, subnets, Cloud NAT
    ├── gke/                    # GKE Autopilot cluster
    ├── cloud-sql/              # PostgreSQL database
    ├── storage/                # GCS bucket + Artifact Registry
    └── iam/                    # Service accounts, Workload Identity
```

## Modules

### networking

Creates VPC network with:
- Primary subnet for GKE nodes
- Secondary ranges for pods and services
- Cloud NAT for egress traffic
- Private Google Access enabled

### gke

Creates GKE Autopilot cluster with:
- Workload Identity enabled
- Private cluster (optional)
- VPC-native networking
- Cloud Logging and Monitoring

### cloud-sql

Creates Cloud SQL PostgreSQL instance with:
- Private IP (VPC peering)
- Automated backups
- Configurable machine type
- Database and user creation

### storage

Creates:
- GCS bucket for model storage (with versioning)
- Artifact Registry for container images

### iam

Creates:
- Kubernetes service accounts
- GCP service accounts
- Workload Identity bindings
- IAM roles for GCS and Cloud SQL access

## Configuration

### Environment Variables

After deployment, configure your workloads with these environment variables:

```bash
# Database connection (via Cloud SQL Proxy sidecar)
CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:PASSWORD@localhost:5432/cartridge

# Or via private IP
CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://cartridge:PASSWORD@CLOUD_SQL_IP:5432/cartridge

# GCS for model storage (using S3-compatible API)
CARTRIDGE_STORAGE_MODEL_BACKEND=s3
CARTRIDGE_STORAGE_S3_BUCKET=your-project-cartridge-models
CARTRIDGE_STORAGE_S3_ENDPOINT=https://storage.googleapis.com

# Or use native GCS (requires code changes)
CARTRIDGE_STORAGE_MODEL_BACKEND=gcs
CARTRIDGE_STORAGE_GCS_BUCKET=your-project-cartridge-models
```

### Workload Identity

The IAM module creates service accounts with Workload Identity bindings.
Annotate your Kubernetes service accounts:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cartridge-actor
  annotations:
    iam.gke.io/gcp-service-account: cartridge-actor@PROJECT.iam.gserviceaccount.com
```

## Cost Estimates

### Development

| Resource | Spec | Monthly Cost (est.) |
|----------|------|---------------------|
| GKE Autopilot | ~4 vCPU, 8GB | $50-100 |
| Cloud SQL | db-f1-micro | $10 |
| GCS | 10GB | $0.26 |
| Cloud NAT | 1 gateway | $32 |
| **Total** | | **~$100-150** |

### Production

| Resource | Spec | Monthly Cost (est.) |
|----------|------|---------------------|
| GKE Autopilot | ~16 vCPU, 32GB | $200-400 |
| Cloud SQL | db-custom-2-4096 | $50 |
| GCS | 100GB | $2.60 |
| Cloud NAT | 1 gateway | $32 |
| Load Balancer | 1 forwarding rule | $18 |
| **Total** | | **~$300-500** |

## Cleanup

```bash
# Destroy all resources
cd terraform/environments/dev
terraform destroy

# Or just specific resources
terraform destroy -target=module.gke
```

## Troubleshooting

### Cloud SQL Connection Issues

1. Ensure the VPC peering is established:
   ```bash
   gcloud services vpc-peerings list --network=cartridge-vpc
   ```

2. Check Cloud SQL has private IP:
   ```bash
   gcloud sql instances describe cartridge-postgres --format="value(ipAddresses)"
   ```

### GKE Workload Identity Issues

1. Verify the binding exists:
   ```bash
   gcloud iam service-accounts get-iam-policy cartridge-actor@PROJECT.iam.gserviceaccount.com
   ```

2. Check pod identity:
   ```bash
   kubectl exec -it POD_NAME -- curl -H "Metadata-Flavor: Google" \
     http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email
   ```

### Terraform State

For team collaboration, configure remote state:

```hcl
terraform {
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "cartridge/dev"
  }
}
```
