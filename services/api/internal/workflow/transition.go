// Package workflow holds the pure state machines of the product rules:
// the description editor gate and the ingest/generation job lifecycles.
package workflow

// Description job statuses (db: generation_jobs.status).
const (
	JobQueued     = "queued"
	JobRunning    = "running"
	JobDraftReady = "draft_ready"
	JobApproved   = "approved"
	JobRejected   = "rejected"
	JobPublished  = "published"
	JobFailed     = "failed"
)

// Review decisions (db: editor_reviews.decision).
const (
	DecisionApprove = "approve"
	DecisionReject  = "reject"
	DecisionEdit    = "edit"
)

// Ingest job statuses.
const (
	IngestQueued    = "queued"
	IngestRunning   = "running"
	IngestSucceeded = "succeeded"
	IngestFailed    = "failed"
)

// CanReview reports whether a human decision is allowed in the current status.
// Editor gate: only a finished draft can be reviewed.
func CanReview(status, decision string) bool {
	if status != JobDraftReady {
		return false
	}
	switch decision {
	case DecisionApprove, DecisionReject, DecisionEdit:
		return true
	default:
		return false
	}
}

// ReviewOutcome maps (status, decision) to the next status.
func ReviewOutcome(decision string) (string, bool) {
	switch decision {
	case DecisionApprove, DecisionEdit:
		return JobApproved, true
	case DecisionReject:
		return JobRejected, true
	default:
		return "", false
	}
}

// CanPublish enforces that only approved descriptions reach the client.
func CanPublish(status string) bool { return status == JobApproved }

// CanTransition is the generic job progress guard for workers.
func CanTransition(from, to string) bool {
	allowed := map[string][]string{
		JobQueued:     {JobRunning, JobFailed},
		JobRunning:    {JobDraftReady, JobFailed},
		JobDraftReady: {JobApproved, JobRejected},
		JobApproved:   {JobPublished},
		JobRejected:   {},
		JobPublished:  {},
		JobFailed:     {},
	}
	for _, next := range allowed[from] {
		if next == to {
			return true
		}
	}
	return false
}
