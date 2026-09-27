package workflow

import "testing"

func TestCanReview(t *testing.T) {
	cases := []struct {
		status, decision string
		want             bool
	}{
		{JobDraftReady, DecisionApprove, true},
		{JobDraftReady, DecisionEdit, true},
		{JobDraftReady, DecisionReject, true},
		{JobDraftReady, "nonsense", false},
		{JobQueued, DecisionApprove, false},
		{JobRunning, DecisionApprove, false},
		{JobApproved, DecisionReject, false},
		{JobPublished, DecisionEdit, false},
		{JobFailed, DecisionApprove, false},
	}
	for _, c := range cases {
		if got := CanReview(c.status, c.decision); got != c.want {
			t.Errorf("CanReview(%s,%s)=%v want %v", c.status, c.decision, got, c.want)
		}
	}
}

func TestReviewOutcome(t *testing.T) {
	if s, ok := ReviewOutcome(DecisionApprove); !ok || s != JobApproved {
		t.Errorf("approve -> %s %v", s, ok)
	}
	if s, ok := ReviewOutcome(DecisionEdit); !ok || s != JobApproved {
		t.Errorf("edit -> %s %v", s, ok)
	}
	if s, ok := ReviewOutcome(DecisionReject); !ok || s != JobRejected {
		t.Errorf("reject -> %s %v", s, ok)
	}
	if _, ok := ReviewOutcome("publish"); ok {
		t.Error("unknown decision must not map to a status")
	}
}

func TestEditorGate(t *testing.T) {
	if CanPublish(JobDraftReady) {
		t.Error("draft_ready must not be publishable (editor gate)")
	}
	if CanPublish(JobApproved) != true {
		t.Error("approved must be publishable")
	}
	for _, s := range []string{JobQueued, JobRunning, JobRejected, JobPublished, JobFailed} {
		if CanPublish(s) {
			t.Errorf("%s must not be publishable", s)
		}
	}
}

func TestCanTransition(t *testing.T) {
	ok := [][2]string{{JobQueued, JobRunning}, {JobRunning, JobDraftReady},
		{JobDraftReady, JobApproved}, {JobApproved, JobPublished},
		{JobRunning, JobFailed}, {JobQueued, JobFailed}}
	for _, pair := range ok {
		if !CanTransition(pair[0], pair[1]) {
			t.Errorf("%s -> %s should be allowed", pair[0], pair[1])
		}
	}
	bad := [][2]string{{JobQueued, JobPublished}, {JobPublished, JobDraftReady},
		{JobFailed, JobRunning}, {JobApproved, JobDraftReady}}
	for _, pair := range bad {
		if CanTransition(pair[0], pair[1]) {
			t.Errorf("%s -> %s should be rejected", pair[0], pair[1])
		}
	}
}
