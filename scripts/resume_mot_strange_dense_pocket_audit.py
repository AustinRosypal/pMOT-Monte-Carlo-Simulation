"""Resume immutable witness replays with Windows sharing-violation retries."""

import audit_mot_strange_dense_pocket_recovery as auditor
from resume_mot_strange_sampled_production import write_with_retry


if __name__ == "__main__":
    auditor.write_json = write_with_retry
    auditor.audit()
