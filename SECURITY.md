# Security Policy

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report vulnerabilities privately. Do not open a public issue for a
security problem.

Use GitHub's private vulnerability reporting on the repository's **Security**
tab by selecting **Report a vulnerability**. This keeps the details private
while the issue is investigated.

You can expect an initial response within 7 days.

## Security Measures

- Secrets are never stored in the repository. Configuration is documented with
  `.env.example`, which contains only placeholder values.
- PDF ingestion enforces limits on file size, file type (PDF magic bytes), and
  page count, and rejects symlinks and encrypted files.
- Index metadata is stored as JSON instead of pickle, so loading an index never
  unpickles data.
- CI runs `bandit` for static security analysis and `pip-audit` against the
  installed environment.
- Dependabot checks for dependency and GitHub Actions updates weekly.
