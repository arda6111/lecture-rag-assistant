# Security Policy

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please report vulnerabilities privately and do not open a public issue for a
security problem.

Use GitHub private vulnerability reporting: open the repository's Security tab
and choose "Report a vulnerability". This option becomes available once the
repository is public. Until then there is no private reporting channel, so
please hold back details until it is available.

Include the affected version, steps to reproduce, and the impact you expect.

## Security Measures

- Secrets are never stored in the repository. Configuration is documented with
  `.env.example`, which contains only placeholder values.
- PDF ingestion enforces limits on file size, file type (PDF magic bytes), and
  page count, and rejects symlinks and encrypted files (including PDFs with
  only an owner password).
- Index metadata is stored as JSON instead of pickle, so loading an index never
  unpickles data.
- CI runs `bandit` for static security analysis and `pip-audit` against the
  installed environment.
- Dependabot checks for dependency and GitHub Actions updates weekly.
