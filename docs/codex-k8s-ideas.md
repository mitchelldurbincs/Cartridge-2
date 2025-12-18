# Kubernetes Adoption To-Do List

A working checklist for moving this project toward Kubernetes locally and in the cloud.

## Foundations
- Container image hardening: add image labels, non-root user, minimal base image, healthcheck, and multi-arch build support.
- Configuration strategy: standardize on environment variables + config files; add default values and validation; document required secrets.
- Observability: standardize structured logging, add request/worker metrics with Prometheus format, and add basic traces where possible.
- Dependency review: prune unused dependencies, pin versions, and add vulnerability scanning (e.g., Trivy, Snyk) to CI.

## Local Kubernetes
- Create a local dev manifest set (kind or k3d) with Namespaces, Deployments, Services, and Ingress for web/engine/trainer.
- Provide Helm chart or Kustomize overlays for local iteration; include values for ports, image tags, and resource requests/limits.
- Add make targets for `kind create cluster`, image load, and `kubectl` apply/delete for quick developer onboarding.
- Integrate Tilt or Skaffold for live-reload and rapid image rebuilds during development.

## Cloud Readiness
- Define production-grade manifests (Helm/Kustomize) with pod disruption budgets, autoscaling (HPA), pod security contexts, and network policies.
- Secret management: integrate with external secret store (e.g., AWS Secrets Manager/GCP Secret Manager) via CSI driver.
- Storage: document persistent volume needs (size, access mode, class) for any stateful components.
- Ingress: select ingress controller (NGINX/Traefik/Gateway API) and configure TLS termination and redirects.
- Scheduling: set topology spread constraints/affinity for multi-AZ availability; enable readiness/liveness/startup probes.
- Cost controls: set resource quotas and request/limit defaults per namespace; schedule non-critical jobs to spot/preemptible nodes.

## CI/CD (GitHub Actions)
- Build pipeline: lint, format, type-check, and run tests; build and scan container images; push to registry with tags (`sha`, `semver`).
- Manifest validation: run `kubeconform`/`kubectl apply --dry-run` on manifests; template-check Helm charts.
- Deploy pipeline: environment-specific workflows triggered on tags or branches; use GitHub Environments with required approvals; run canary/blue-green when feasible.
- Automated rollbacks: record deployment status and enable rollback on failed health checks.
- Release artifacts: publish Helm chart/package and changelog; attach SBOM and provenance (SLSA) for supply-chain integrity.

## Operational Readiness
- Runbooks: document incident playbooks (restart pods, rollbacks, scaling, log/metric dashboards).
- Backups: define backup/restore plan for any data stores and schedule regular jobs.
- DR/HA: plan for multi-region failover if relevant; test cluster restore procedures.
- Access: role-based access control (RBAC) per team/function; audit logging enabled.

## Cleanup/Refactoring for Cloud Efficiency
- Split monolithic containers if needed (web vs. worker) to scale independently; remove unused build artifacts.
- Right-size resources: profile CPU/memory; set conservative requests/limits and revisit after load testing.
- Add feature flags to disable expensive/experimental paths in cloud deployments.
- Optimize images: multi-stage builds, cache dependencies, and remove dev tools from runtime layers.
- Network efficiency: enable gzip/brotli on ingress, connection pooling, and keep-alive where applicable.
- Job scheduling: batch/cron workloads off-peak; consolidate cron jobs into a single CronJob controller with concurrencyPolicy.

## Milestones
- Milestone 1: Local kind cluster with working deployments, ingress, and CI lint/test/build pipeline.
- Milestone 2: Staging namespace in cloud with GitHub Actions deploy, secrets via CSI, and HPA configured.
- Milestone 3: Production-ready hardening—network policies, PDBs, canary/blue-green rollout, monitoring dashboards, and runbooks in place.
- Milestone 4: Cost optimization cycle—resource tuning from observed metrics, spot/preemptible rollout for non-critical workloads, and storage/egress reviews.