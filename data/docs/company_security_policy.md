# Enterprise Security and Data Protection Policy

## 1. Authentication and Access Control
All employees and contractors must authenticate using multi-factor authentication (MFA) backed by FIDO2 hardware security keys. Single-sign-on (SSO) sessions expire automatically after 8 hours of inactivity. Privileged production access requires Just-In-Time (JIT) role elevation with approval from an engineering team lead.

## 2. Encryption Standards
Data at rest must be encrypted using AES-256-GCM algorithms with customer-managed encryption keys (CMEK) rotated every 90 days. All data in transit across public and internal network boundaries must utilize TLS 1.3 with mutual authentication (mTLS). Unencrypted HTTP traffic is prohibited across all endpoints.

## 3. Data Retention and Destruction
Customer audit logs must be retained in immutable cold storage for exactly 7 years to comply with SOC-2 Type II standards. Following the retention window, data must be cryptographically wiped in accordance with NIST SP 800-88 guidelines.

## 4. Incident Response Timeframes
Any suspected data breach or unauthorized access must be reported to the Security Operations Center (SOC) within 15 minutes of detection. The incident commander must publish an initial status update to executive stakeholders within 60 minutes.
