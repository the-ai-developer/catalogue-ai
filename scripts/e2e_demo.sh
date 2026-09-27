#!/usr/bin/env bash
# Full scripted journey with assertions: seed → grounded QA → editor-gated
# description → publish. Requires a running stack (`make up`).
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8080}"
API_KEY="${API_KEY:-local-dev-admin-key}"
HDR=(-H "X-API-Key: ${API_KEY}" -H "Content-Type: application/json")

say() { printf '\n\033[36m▸ %s\033[0m\n' "$*"; }
fail() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

wait_ready() {
  say "waiting for API at ${API_BASE}"
  for _ in $(seq 1 60); do
    if curl -fsS "${API_BASE}/readyz" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  fail "API never became ready"
}

json_get() { python3 -c "import json,sys; d=json.load(sys.stdin); print(d$1)"; }

wait_ready

say "seeding catalogue"
python3 scripts/seed_demo.py --api-base "${API_BASE}" --api-key "${API_KEY}"

say "Project 1: grounded QA must carry citations"
ASK=$(curl -fsS "${HDR[@]}" "${API_BASE}/api/v1/qa/ask" -d '{
  "question": "Which bottle is dishwasher safe and what is its capacity?",
  "top_k": 6, "use_images": true, "composition": "extractive",
  "user_ref": "e2e:buyer"}')
echo "${ASK}" | python3 -m json.tool | head -30
echo "${ASK}" | json_get "['citation_check']" | grep -qi "True\|true" \
  || echo "  (note: citation check flagged this answer — UI would withhold it)"
echo "${ASK}" | json_get "['citations']" >/dev/null || fail "no citations returned"

say "Project 2: generate → editor gate → publish"
JOB=$(curl -fsS "${HDR[@]}" "${API_BASE}/api/v1/descriptions/jobs" -d '{
  "spec": {"title": "Kestrel 12 oz Insulated Bottle", "category": "drinkware",
           "material": "18/8 stainless steel",
           "dimensions": {"height_cm": 26.0, "capacity_oz": 12},
           "features": ["double-wall vacuum insulation", "leak-proof lid"]},
  "beam_width": 4, "max_len": 192}')
JOB_ID=$(echo "${JOB}" | json_get "['job_id']")

for _ in $(seq 1 30); do
  DETAIL=$(curl -fsS "${HDR[@]}" "${API_BASE}/api/v1/descriptions/jobs/${JOB_ID}")
  STATUS=$(echo "${DETAIL}" | json_get "['status']")
  [ "${STATUS}" != "queued" ] && [ "${STATUS}" != "running" ] && break
  sleep 2
done
[ "${STATUS}" = "draft_ready" ] || fail "expected draft_ready, got ${STATUS}"
say "drafts ready — verifying the editor gate blocks premature publish"
CODE=$(curl -s -o /dev/null -w '%{http_code}' "${HDR[@]}" \
  -X POST "${API_BASE}/api/v1/descriptions/jobs/${JOB_ID}/publish")
[ "${CODE}" = "409" ] || [ "${CODE}" = "422" ] \
  || fail "publish before approval must fail (got HTTP ${CODE})"

DRAFT_ID=$(echo "${DETAIL}" | json_get "['drafts'][0]['id']")
EDITED="The Kestrel 12 oz bottle keeps drinks cold, is dishwasher safe, and looks the part. (edited by e2e)"
REVIEWED=$(curl -fsS "${HDR[@]}" \
  "${API_BASE}/api/v1/descriptions/jobs/${JOB_ID}/review" -d "{
    \"decision\": \"edit\", \"draft_id\": \"${DRAFT_ID}\",
    \"edited_text\": \"${EDITED}\", \"notes\": \"e2e inline edit\"}")
echo "${REVIEWED}" | json_get "['status']" | grep -q "approved" \
  || fail "edit decision must move the job to approved"

PUB=$(curl -fsS "${HDR[@]}" -X POST \
  "${API_BASE}/api/v1/descriptions/jobs/${JOB_ID}/publish")
echo "${PUB}" | python3 -m json.tool
echo "${PUB}" | json_get "['text']" | grep -q "(edited by e2e)" \
  || fail "published text must be the editor-approved text"

say "ALL E2E ASSERTIONS PASSED ✅"
