# Security Policy

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please report vulnerabilities privately. Do not open a public issue for a
security problem.

Contact the maintainer through the GitHub profile
[@arda6111](https://github.com/arda6111) and ask for a private channel
before sharing details. Include the affected version, steps to reproduce,
and the impact you expect.

You can expect an initial response within 7 days.

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
