# Cloud Infrastructure and Service Architecture Spec

## 1. High Availability and SLA Guarantees
The microservice cluster is deployed across three distinct availability zones (AZs) in the us-central1 region. The platform guarantees an uptime Service Level Agreement (SLA) of 99.99% availability per calendar quarter. Any planned maintenance must be scheduled during the designated maintenance window between Sunday 02:00 UTC and 04:00 UTC.

## 2. Autoscaling and Load Balancing
The Kubernetes pod autoscaler (HPA) monitors both 95th percentile latency (target <= 250ms) and CPU utilization (target <= 70%). Scaling actions trigger within 30 seconds of sustained threshold violations, scaling from a minimum baseline of 6 replicas to a burst ceiling of 120 replicas.

## 3. Database Replication and Backups
The primary PostgreSQL instance streams write-ahead logs (WAL) synchronously to a hot standby in AZ-2 and asynchronously to a read replica in AZ-3. Automated point-in-time recovery (PITR) snapshots are captured hourly and archived to multi-region object storage with object locking enabled.

## 4. Rate Limiting and Traffic Throttling
Inbound public API endpoints enforce token-bucket rate limiting calibrated to 1,000 requests per minute per API client key, with a burst allowance of 200 requests. Requests violating quotas receive HTTP status code 429 Too Many Requests with a Retry-After header.
