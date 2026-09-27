// Package config loads the Go API configuration from environment variables.
// Names match catalogue-ai/.env.example exactly (plus MIGRATIONS_DIR, which
// controls where the SQL migration runner looks for *.sql files).
package config

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// BootstrapKey is a key parsed from BOOTSTRAP_API_KEYS (format
// "name:sha256hex:role", comma separated). The DB `api_keys` table is
// authoritative; bootstrap keys are a fallback for first boot / local dev.
type BootstrapKey struct {
	Name    string
	KeyHash string // sha256 hex of the secret
	Role    string // viewer | editor | admin
}

// Config is the full runtime configuration for the API service.
type Config struct {
	DatabaseURL        string
	HTTPAddr           string
	ModelServerURL     string
	AssetStorageDir    string
	AssetPublicBaseURL string
	ModelTimeoutShort  time.Duration
	ModelTimeoutLong   time.Duration
	WorkerPollInterval time.Duration
	LogLevel           string
	BootstrapAPIKeys   []BootstrapKey
	MigrationsDir      string
	MaxChunkTokens     int
	ChunkOverlapTokens int
}

// Load reads configuration from the environment, applying defaults that are
// suitable for local development.
func Load() (*Config, error) {
	cfg := &Config{
		DatabaseURL:        getenv("DATABASE_URL", "postgres://catalogue:catalogue@localhost:5432/catalogue?sslmode=disable"),
		HTTPAddr:           getenv("HTTP_ADDR", ":8080"),
		ModelServerURL:     strings.TrimRight(getenv("MODEL_SERVER_URL", "http://localhost:8090"), "/"),
		AssetStorageDir:    getenv("ASSET_STORAGE_DIR", "/data/assets"),
		AssetPublicBaseURL: strings.TrimRight(getenv("ASSET_PUBLIC_BASE_URL", "/assets"), "/"),
		LogLevel:           getenv("LOG_LEVEL", "info"),
		MigrationsDir:      getenv("MIGRATIONS_DIR", "db/migrations"),
		MaxChunkTokens:     getint("MAX_CHUNK_TOKENS", 220),
		ChunkOverlapTokens: getint("CHUNK_OVERLAP_TOKENS", 40),
	}

	var err error
	if cfg.ModelTimeoutShort, err = getduration("MODEL_TIMEOUT_SHORT", 5*time.Second); err != nil {
		return nil, err
	}
	if cfg.ModelTimeoutLong, err = getduration("MODEL_TIMEOUT_LONG", 30*time.Second); err != nil {
		return nil, err
	}
	if cfg.WorkerPollInterval, err = getduration("WORKER_POLL_INTERVAL", 2*time.Second); err != nil {
		return nil, err
	}

	if cfg.BootstrapAPIKeys, err = parseBootstrapKeys(os.Getenv("BOOTSTRAP_API_KEYS")); err != nil {
		return nil, err
	}
	return cfg, nil
}

func getenv(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}

func getduration(name string, def time.Duration) (time.Duration, error) {
	v := os.Getenv(name)
	if v == "" {
		return def, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("%s: invalid duration %q: %w", name, v, err)
	}
	if d <= 0 {
		return 0, fmt.Errorf("%s: duration must be positive, got %q", name, v)
	}
	return d, nil
}

func getint(name string, def int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return def
	}
	var v int
	if _, err := fmt.Sscanf(raw, "%d", &v); err != nil {
		return def
	}
	return v
}

func parseBootstrapKeys(raw string) ([]BootstrapKey, error) {
	var keys []BootstrapKey
	for _, entry := range strings.Split(raw, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		parts := strings.Split(entry, ":")
		if len(parts) != 3 {
			return nil, fmt.Errorf("BOOTSTRAP_API_KEYS: invalid entry %q (want name:sha256hex:role)", entry)
		}
		name, hash, role := parts[0], strings.ToLower(parts[1]), strings.ToLower(parts[2])
		if name == "" || hash == "" {
			return nil, fmt.Errorf("BOOTSTRAP_API_KEYS: invalid entry %q (empty name or hash)", entry)
		}
		switch role {
		case "viewer", "editor", "admin":
		default:
			return nil, fmt.Errorf("BOOTSTRAP_API_KEYS: invalid role %q in entry %q (want viewer|editor|admin)", role, entry)
		}
		keys = append(keys, BootstrapKey{Name: name, KeyHash: hash, Role: role})
	}
	return keys, nil
}
