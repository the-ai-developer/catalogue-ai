package httpapi

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"catalogue-ai/services/api/internal/config"
)

func TestHashKey(t *testing.T) {
	if HashKey("secret") != HashKey("secret") {
		t.Fatal("hash must be deterministic")
	}
	if HashKey("secret") == HashKey("other") {
		t.Fatal("different keys must hash differently")
	}
}

func TestRequireRoles(t *testing.T) {
	keys := staticKeys{
		HashKey("viewer-key"): "viewer",
		HashKey("editor-key"): "editor",
		HashKey("admin-key"):  "admin",
	}
	ok := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(200)
	})

	cases := []struct {
		name, key, needed string
		want              int
	}{
		{"no key", "", "viewer", 401},
		{"unknown key", "bogus", "viewer", 401},
		{"viewer reads", "viewer-key", "viewer", 200},
		{"viewer blocked from admin", "viewer-key", "admin", 403},
		{"editor on editor route", "editor-key", "editor", 200},
		{"editor blocked from admin", "editor-key", "admin", 403},
		{"admin does everything", "admin-key", "admin", 200},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/api/v1/items", nil)
			if c.key != "" {
				req.Header.Set("X-API-Key", c.key)
			}
			rec := httptest.NewRecorder()
			Require(keys, c.needed)(ok).ServeHTTP(rec, req)
			if rec.Code != c.want {
				t.Fatalf("status=%d want %d body=%s", rec.Code, c.want, rec.Body.String())
			}
		})
	}
}

func TestBootstrapKeysFallback(t *testing.T) {
	cfg := &config.Config{BootstrapAPIKeys: []config.BootstrapKey{
		{Name: "local", KeyHash: HashKey("boot"), Role: "admin"}}}
	chain := Chain{staticKeys{}, BootstrapKeys(cfg)}
	if role, ok := chain.RoleForKeyHash(HashKey("boot")); !ok || role != "admin" {
		t.Fatalf("bootstrap key not resolved: %s %v", role, ok)
	}
	if _, ok := chain.RoleForKeyHash("nope"); ok {
		t.Fatal("unknown key must not resolve")
	}
}
