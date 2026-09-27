package httpapi

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"

	"catalogue-ai/services/api/internal/apperr"
	"catalogue-ai/services/api/internal/config"
)

// Role ranks: viewer < editor < admin.
var roleRank = map[string]int{"viewer": 1, "editor": 2, "admin": 3}

// KeyVerifier resolves an API key to a role (DB-backed, with bootstrap fallback).
type KeyVerifier interface {
	RoleForKeyHash(hash string) (string, bool)
}

type staticKeys map[string]string // sha256hex -> role

func (s staticKeys) RoleForKeyHash(hash string) (string, bool) {
	role, ok := s[hash]
	return role, ok
}

// BootstrapKeys turns BOOTSTRAP_API_KEYS entries (name:sha256hex:role) into a
// verifier used before/alongside the api_keys table.
func BootstrapKeys(cfg *config.Config) staticKeys {
	out := staticKeys{}
	for _, k := range cfg.BootstrapAPIKeys {
		out[k.KeyHash] = k.Role
	}
	return out
}

// HashKey returns the sha256 hex digest stored for an API key.
func HashKey(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

// Chain tries verifiers in order (DB first, then bootstrap).
type Chain []KeyVerifier

func (c Chain) RoleForKeyHash(hash string) (string, bool) {
	for _, v := range c {
		if role, ok := v.RoleForKeyHash(hash); ok {
			return role, true
		}
	}
	return "", false
}

// Require returns middleware enforcing at least `needed` role.
func Require(v KeyVerifier, needed string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			key := r.Header.Get("X-API-Key")
			if key == "" {
				writeErr(w, apperr.UnauthorizedErr("missing X-API-Key"))
				return
			}
			role, ok := v.RoleForKeyHash(HashKey(key))
			if !ok {
				writeErr(w, apperr.UnauthorizedErr("unknown API key"))
				return
			}
			if roleRank[role] < roleRank[needed] {
				writeErr(w, apperr.ForbiddenErr("role "+role+" < "+needed))
				return
			}
			next.ServeHTTP(w, r)
		}))
	}
}
