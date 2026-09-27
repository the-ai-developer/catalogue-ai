package db

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed migrations/*.sql
var embeddedMigrations embed.FS

// Migrate applies all pending *.sql migrations from dir (default behaviour:
// read dir from disk; if it does not exist, use the embedded copy). Each file
// is applied once inside a transaction and recorded in schema_migrations.
// Files may use the `-- +migrate Up` / `-- +migrate Down` markers; only the
// Up section is executed.
func Migrate(ctx context.Context, pool *pgxpool.Pool, dir string) error {
	if dir == "" {
		dir = "db/migrations"
	}
	files, err := loadMigrations(dir)
	if err != nil {
		return err
	}

	if _, err := pool.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS schema_migrations (
			name       text PRIMARY KEY,
			applied_at timestamptz NOT NULL DEFAULT now()
		)`); err != nil {
		return fmt.Errorf("create schema_migrations: %w", err)
	}

	applied := map[string]bool{}
	rows, err := pool.Query(ctx, `SELECT name FROM schema_migrations`)
	if err != nil {
		return fmt.Errorf("read schema_migrations: %w", err)
	}
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			rows.Close()
			return fmt.Errorf("scan schema_migrations: %w", err)
		}
		applied[name] = true
	}
	rows.Close()
	if rows.Err() != nil {
		return fmt.Errorf("read schema_migrations: %w", rows.Err())
	}

	for _, f := range files {
		if applied[f.name] {
			continue
		}
		sql := upSection(f.body)
		if strings.TrimSpace(sql) == "" {
			continue
		}
		tx, err := pool.Begin(ctx)
		if err != nil {
			return fmt.Errorf("begin migration %s: %w", f.name, err)
		}
		if _, err := tx.Exec(ctx, sql); err != nil {
			tx.Rollback(ctx)
			return fmt.Errorf("apply migration %s: %w", f.name, err)
		}
		if _, err := tx.Exec(ctx, `INSERT INTO schema_migrations (name) VALUES ($1)`, f.name); err != nil {
			tx.Rollback(ctx)
			return fmt.Errorf("record migration %s: %w", f.name, err)
		}
		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("commit migration %s: %w", f.name, err)
		}
	}
	return nil
}

type migrationFile struct {
	name string
	body string
}

func loadMigrations(dir string) ([]migrationFile, error) {
	entries, err := os.ReadDir(dir)
	if err == nil {
		var files []migrationFile
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".sql") {
				continue
			}
			body, err := os.ReadFile(filepath.Join(dir, e.Name()))
			if err != nil {
				return nil, fmt.Errorf("read migration %s: %w", e.Name(), err)
			}
			files = append(files, migrationFile{name: e.Name(), body: string(body)})
		}
		sort.Slice(files, func(i, j int) bool { return files[i].name < files[j].name })
		return files, nil
	}
	if !os.IsNotExist(err) {
		return nil, fmt.Errorf("read migrations dir %s: %w", dir, err)
	}

	// Fallback: embedded copy.
	var files []migrationFile
	err = fs.WalkDir(embeddedMigrations, "migrations", func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".sql") {
			return err
		}
		body, err := embeddedMigrations.ReadFile(path)
		if err != nil {
			return err
		}
		files = append(files, migrationFile{name: filepath.Base(path), body: string(body)})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("read embedded migrations: %w", err)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].name < files[j].name })
	return files, nil
}

// upSection extracts the `-- +migrate Up` section of a migration file. Files
// without markers are executed verbatim.
func upSection(body string) string {
	const up = "-- +migrate Up"
	const down = "-- +migrate Down"
	if i := strings.Index(body, up); i >= 0 {
		body = body[i+len(up):]
		if j := strings.Index(body, down); j >= 0 {
			body = body[:j]
		}
	} else if j := strings.Index(body, down); j >= 0 {
		body = body[:j]
	}
	return body
}
