// Package store is the PostgreSQL data layer (schema: db/migrations).
package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct{ Pool *pgxpool.Pool }

func New(pool *pgxpool.Pool) *Store { return &Store{Pool: pool} }

var ErrNotFound = errors.New("not found")

// ------------------------------- domain types -------------------------------

type Counts struct {
	Chunks          int `json:"chunks"`
	EmbeddingsText  int `json:"embeddings_text"`
	EmbeddingsImage int `json:"embeddings_image"`
}

type Asset struct {
	ID        string    `json:"id"`
	Kind      string    `json:"kind"`
	URL       string    `json:"url"`
	Mime      string    `json:"mime"`
	Width     *int      `json:"width"`
	Height    *int      `json:"height"`
	Bytes     int64     `json:"bytes"`
	SHA256    string    `json:"sha256"`
	CreatedAt time.Time `json:"created_at"`
}

type Item struct {
	ID         string         `json:"id"`
	SKU        string         `json:"sku"`
	Title      string         `json:"title"`
	Category   string         `json:"category"`
	Material   *string        `json:"material"`
	Dimensions map[string]any `json:"dimensions"`
	Features   []string       `json:"features"`
	Extra      map[string]any `json:"extra"`
	Status     string         `json:"status"`
	CreatedAt  time.Time      `json:"created_at"`
	UpdatedAt  time.Time      `json:"updated_at"`
	Assets     []Asset        `json:"assets"`
	Counts     Counts         `json:"counts"`
}

type IngestJob struct {
	ID            string     `json:"id"`
	ItemID        string     `json:"item_id"`
	Status        string     `json:"status"`
	ChunksIndexed int        `json:"chunks_indexed"`
	ImagesIndexed int        `json:"images_indexed"`
	Error         *string    `json:"error"`
	CreatedAt     time.Time  `json:"created_at"`
	StartedAt     *time.Time `json:"started_at"`
	FinishedAt    *time.Time `json:"finished_at"`
}

type ChunkRow struct {
	ID      string
	Text    string
	Ordinal int
}

type Query struct {
	ID           string    `json:"id"`
	UserRef      string    `json:"user_ref"`
	Question     string    `json:"question"`
	ScopeItemIDs []string  `json:"scope_item_ids"`
	UseImages    bool      `json:"use_images"`
	TopK         int       `json:"top_k"`
	CreatedAt    time.Time `json:"created_at"`
}

type CitationRow struct {
	SentenceIndex int     `json:"sentence_index"`
	ItemID        string  `json:"item_id"`
	SKU           string  `json:"sku"`
	ChunkID       *string `json:"chunk_id"`
	AssetID       *string `json:"asset_id"`
	Modality      string  `json:"modality"`
	Score         float64 `json:"score"`
	Snippet       string  `json:"snippet"`
}

type Answer struct {
	ID                string         `json:"answer_id"`
	QueryID           string         `json:"query_id"`
	Answer            string         `json:"answer"`
	Composition       string         `json:"composition"`
	ModelVersion      string         `json:"model_version"`
	CitationCheck     map[string]any `json:"citation_check"`
	CitationCheckPass bool           `json:"-"`
	LatencyMS         int            `json:"latency_ms"`
	CreatedAt         time.Time      `json:"created_at"`
	Citations         []CitationRow  `json:"citations"`
}

type SpecSheet struct {
	ID         string         `json:"id"`
	ItemID     *string        `json:"item_id"`
	Title      *string        `json:"title"`
	Category   string         `json:"category"`
	Material   *string        `json:"material"`
	Dimensions map[string]any `json:"dimensions"`
	Features   []string       `json:"features"`
	Extra      map[string]any `json:"extra"`
	CreatedBy  string         `json:"created_by"`
	CreatedAt  time.Time      `json:"created_at"`
}

type Draft struct {
	ID    string  `json:"id"`
	JobID string  `json:"job_id"`
	Rank  int     `json:"rank"`
	Text  string  `json:"text"`
	Score float64 `json:"score"`
}

type Review struct {
	ID         string    `json:"id"`
	JobID      string    `json:"job_id"`
	DraftID    *string   `json:"draft_id"`
	Reviewer   string    `json:"reviewer"`
	Decision   string    `json:"decision"`
	EditedText *string   `json:"edited_text"`
	Notes      *string   `json:"notes"`
	CreatedAt  time.Time `json:"created_at"`
}

type Published struct {
	ID         string    `json:"id"`
	ItemID     string    `json:"item_id"`
	JobID      string    `json:"job_id"`
	Text       string    `json:"text"`
	ApprovedBy string    `json:"approved_by"`
	CreatedAt  time.Time `json:"created_at"`
}

type GenerationJob struct {
	ID           string      `json:"id"`
	SpecID       string      `json:"spec_id"`
	Status       string      `json:"status"`
	ModelName    *string     `json:"model_name"`
	ModelVersion *string     `json:"model_version"`
	BeamWidth    int         `json:"beam_width"`
	MaxLen       int         `json:"max_len"`
	Error        *string     `json:"error"`
	CreatedAt    time.Time   `json:"created_at"`
	StartedAt    *time.Time  `json:"started_at"`
	FinishedAt   *time.Time  `json:"finished_at"`
	Spec         *SpecSheet  `json:"spec,omitempty"`
	Drafts       []Draft     `json:"drafts"`
	Reviews      []Review    `json:"reviews"`
	Published    []Published `json:"published"`
}

// ------------------------------- helpers ------------------------------------

func marshal(v any) []byte {
	if v == nil {
		return []byte("{}")
	}
	b, err := json.Marshal(v)
	if err != nil {
		return []byte("{}")
	}
	return b
}

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// ------------------------------- items --------------------------------------

func (s *Store) CreateItem(ctx context.Context, it Item) (Item, error) {
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO items (sku, title, category, material, dimensions, features, extra, status)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
		RETURNING id, created_at, updated_at`,
		it.SKU, it.Title, it.Category, it.Material, marshal(it.Dimensions),
		marshal(it.Features), marshal(it.Extra), it.Status).
		Scan(&it.ID, &it.CreatedAt, &it.UpdatedAt)
	if err != nil {
		if strings.Contains(err.Error(), "duplicate key") {
			return it, fmt.Errorf("sku already exists: %w", err)
		}
		return it, err
	}
	return s.GetItem(ctx, it.ID)
}

func (s *Store) GetItem(ctx context.Context, id string) (Item, error) {
	var it Item
	var dims, feats, extra []byte
	err := s.Pool.QueryRow(ctx, `
		SELECT id, sku, title, category, material, dimensions::text, features::text,
		       extra::text, status, created_at, updated_at
		FROM items WHERE id=$1 AND deleted_at IS NULL`, id).
		Scan(&it.ID, &it.SKU, &it.Title, &it.Category, &it.Material, &dims,
			&feats, &extra, &it.Status, &it.CreatedAt, &it.UpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return it, ErrNotFound
	}
	if err != nil {
		return it, err
	}
	_ = json.Unmarshal(dims, &it.Dimensions)
	_ = json.Unmarshal(feats, &it.Features)
	_ = json.Unmarshal(extra, &it.Extra)
	it.Assets, _ = s.ListAssets(ctx, id)
	it.Counts, _ = s.ItemCounts(ctx, id)
	return it, nil
}

type ListFilter struct {
	Query    string
	Category string
	Status   string
	Limit    int
	Cursor   string
}

func (s *Store) ListItems(ctx context.Context, f ListFilter) ([]Item, string, error) {
	if f.Limit <= 0 || f.Limit > 100 {
		f.Limit = 20
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT id, sku, title, category, material, dimensions::text, features::text,
		       extra::text, status, created_at, updated_at
		FROM items
		WHERE deleted_at IS NULL
		  AND ($1='' OR $1 <% sku || ' ' || title || ' ' || category)
		  AND ($2='' OR category=$2)
		  AND ($3='' OR status=$3)
		  AND ($4='' OR id::text > $4)
		ORDER BY id::text
		LIMIT $5`, f.Query, f.Category, f.Status, f.Cursor, f.Limit+1)
	if err != nil {
		return nil, "", err
	}
	defer rows.Close()
	items := []Item{}
	for rows.Next() {
		var it Item
		var dims, feats, extra []byte
		if err := rows.Scan(&it.ID, &it.SKU, &it.Title, &it.Category, &it.Material,
			&dims, &feats, &extra, &it.Status, &it.CreatedAt, &it.UpdatedAt); err != nil {
			return nil, "", err
		}
		_ = json.Unmarshal(dims, &it.Dimensions)
		_ = json.Unmarshal(feats, &it.Features)
		_ = json.Unmarshal(extra, &it.Extra)
		items = append(items, it)
	}
	next := ""
	if len(items) > f.Limit {
		items = items[:f.Limit]
		next = items[len(items)-1].ID
	}
	for i := range items {
		items[i].Assets, _ = s.ListAssets(ctx, items[i].ID)
		items[i].Counts, _ = s.ItemCounts(ctx, items[i].ID)
	}
	return items, next, nil
}

type ItemPatch struct {
	Title      *string
	Category   *string
	Material   *string
	Dimensions map[string]any
	Features   []string
	Extra      map[string]any
	Status     *string
}

func (s *Store) UpdateItem(ctx context.Context, id string, p ItemPatch) (Item, error) {
	_, err := s.Pool.Exec(ctx, `
		UPDATE items SET
		  title      = COALESCE($2, title),
		  category   = COALESCE($3, category),
		  material   = COALESCE($4, material),
		  dimensions = COALESCE($5::jsonb, dimensions),
		  features   = COALESCE($6::jsonb, features),
		  extra      = COALESCE($7::jsonb, extra),
		  status     = COALESCE($8, status)
		WHERE id=$1 AND deleted_at IS NULL`,
		id, p.Title, p.Category, p.Material,
		optionalJSON(p.Dimensions), optionalJSON(p.Features),
		optionalJSON(p.Extra), p.Status)
	if err != nil {
		return Item{}, err
	}
	return s.GetItem(ctx, id)
}

func optionalJSON(v any) []byte {
	if v == nil {
		return nil
	}
	return marshal(v)
}

func (s *Store) SoftDeleteItem(ctx context.Context, id string) error {
	tag, err := s.Pool.Exec(ctx, `
		UPDATE items SET deleted_at=now(), status='archived'
		WHERE id=$1 AND deleted_at IS NULL`, id)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return nil
}

func (s *Store) AddAsset(ctx context.Context, itemID string, a Asset) (Asset, error) {
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO item_assets (item_id, kind, storage_path, mime, width, height, bytes, sha256)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
		RETURNING id, created_at`,
		itemID, a.Kind, a.URL, a.Mime, a.Width, a.Height, a.Bytes, a.SHA256).
		Scan(&a.ID, &a.CreatedAt)
	return a, err
}

func (s *Store) ListAssets(ctx context.Context, itemID string) ([]Asset, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT id, kind, storage_path, mime, width, height, bytes, sha256, created_at
		FROM item_assets WHERE item_id=$1 ORDER BY created_at`, itemID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Asset{}
	for rows.Next() {
		var a Asset
		if err := rows.Scan(&a.ID, &a.Kind, &a.URL, &a.Mime, &a.Width, &a.Height,
			&a.Bytes, &a.SHA256, &a.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, nil
}

func (s *Store) ItemCounts(ctx context.Context, itemID string) (Counts, error) {
	var c Counts
	err := s.Pool.QueryRow(ctx, `
		SELECT
		  (SELECT count(*) FROM chunks WHERE item_id=$1),
		  (SELECT count(*) FROM embeddings WHERE item_id=$1 AND modality='text'),
		  (SELECT count(*) FROM embeddings WHERE item_id=$1 AND modality='image')`,
		itemID).Scan(&c.Chunks, &c.EmbeddingsText, &c.EmbeddingsImage)
	return c, err
}

// ------------------------------- ingest pipeline ----------------------------

func (s *Store) CreateIngestJob(ctx context.Context, itemID string) (string, error) {
	var id string
	err := s.Pool.QueryRow(ctx,
		`INSERT INTO ingest_jobs (item_id) VALUES ($1) RETURNING id`, itemID).Scan(&id)
	return id, err
}

func (s *Store) GetIngestJob(ctx context.Context, id string) (IngestJob, error) {
	var j IngestJob
	err := s.Pool.QueryRow(ctx, `
		SELECT id, item_id, status, chunks_indexed, images_indexed, error,
		       created_at, started_at, finished_at
		FROM ingest_jobs WHERE id=$1`, id).
		Scan(&j.ID, &j.ItemID, &j.Status, &j.ChunksIndexed, &j.ImagesIndexed,
			&j.Error, &j.CreatedAt, &j.StartedAt, &j.FinishedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return j, ErrNotFound
	}
	return j, err
}

// ClaimIngestJob atomically claims the oldest queued job (safe with N replicas).
func (s *Store) ClaimIngestJob(ctx context.Context) (IngestJob, error) {
	var j IngestJob
	err := s.Pool.QueryRow(ctx, `
		UPDATE ingest_jobs SET status='running', started_at=now()
		WHERE id = (SELECT id FROM ingest_jobs WHERE status='queued'
		            ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
		RETURNING id, item_id, status, chunks_indexed, images_indexed, error,
		          created_at, started_at, finished_at`).
		Scan(&j.ID, &j.ItemID, &j.Status, &j.ChunksIndexed, &j.ImagesIndexed,
			&j.Error, &j.CreatedAt, &j.StartedAt, &j.FinishedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return j, ErrNotFound
	}
	return j, err
}

func (s *Store) FinishIngestJob(ctx context.Context, id, status string, chunks,
	images int, errMsg *string) error {
	_, err := s.Pool.Exec(ctx, `
		UPDATE ingest_jobs SET status=$2, chunks_indexed=$3, images_indexed=$4,
		       error=$5, finished_at=now()
		WHERE id=$1`, id, status, chunks, images, errMsg)
	return err
}

// ReplaceChunks swaps the item's chunks in one transaction and returns them.
func (s *Store) ReplaceChunks(ctx context.Context, itemID string, texts []string,
	sources []string, tokenCounts []int) ([]ChunkRow, error) {
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	if _, err := tx.Exec(ctx, `DELETE FROM chunks WHERE item_id=$1`, itemID); err != nil {
		return nil, err
	}
	out := make([]ChunkRow, 0, len(texts))
	for i := range texts {
		var row ChunkRow
		if err := tx.QueryRow(ctx, `
			INSERT INTO chunks (item_id, ordinal, source_kind, text, token_count)
			VALUES ($1,$2,$3,$4,$5) RETURNING id`,
			itemID, i, sources[i], texts[i], tokenCounts[i]).
			Scan(&row.ID); err != nil {
			return nil, err
		}
		row.Text, row.Ordinal = texts[i], i
		out = append(out, row)
	}
	return out, tx.Commit(ctx)
}

// ClearEmbeddings removes an item's embedding rows and returns old faiss ids.
func (s *Store) ClearEmbeddings(ctx context.Context, itemID string) ([]int64, error) {
	rows, err := s.Pool.Query(ctx,
		`DELETE FROM embeddings WHERE item_id=$1 RETURNING coalesce(faiss_id, id)`, itemID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// InsertEmbedding allocates embeddings.id (= faiss_id) for one vector.
func (s *Store) InsertEmbedding(ctx context.Context, itemID, modality string,
	chunkID, assetID *string, model, version string, dim int) (int64, error) {
	var id int64
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO embeddings (item_id, modality, chunk_id, asset_id,
		                        model_name, model_version, dim)
		VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id`,
		itemID, modality, chunkID, assetID, model, version, dim).Scan(&id)
	return id, err
}

func (s *Store) MarkIndexed(ctx context.Context, faissIDs []int64) error {
	if len(faissIDs) == 0 {
		return nil
	}
	_, err := s.Pool.Exec(ctx,
		`UPDATE embeddings SET indexed_at=now() WHERE id = ANY($1)`, faissIDs)
	return err
}

// ------------------------------- QA -----------------------------------------

func (s *Store) InsertQuery(ctx context.Context, q Query) (string, error) {
	scope := "{"
	for i, id := range q.ScopeItemIDs {
		if i > 0 {
			scope += ","
		}
		scope += id
	}
	scope += "}"
	var id string
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO qa_queries (user_ref, question, scope_item_ids, use_images, top_k)
		VALUES ($1,$2,$3::uuid[],$4,$5) RETURNING id`,
		q.UserRef, q.Question, scope, q.UseImages, q.TopK).Scan(&id)
	return id, err
}

func (s *Store) InsertAnswer(ctx context.Context, a Answer) (string, error) {
	var id string
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO qa_answers (query_id, answer_text, composition_mode, model_version,
		                        citation_check_passed, citation_check_detail, latency_ms)
		VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id`,
		a.QueryID, a.Answer, a.Composition, a.ModelVersion,
		a.CitationCheckPass, marshal(a.CitationCheck), a.LatencyMS).Scan(&id)
	return id, err
}

func (s *Store) InsertCitation(ctx context.Context, answerID string, c CitationRow) error {
	_, err := s.Pool.Exec(ctx, `
		INSERT INTO qa_answer_citations
		  (answer_id, sentence_index, item_id, chunk_id, asset_id, modality, score, snippet)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
		answerID, c.SentenceIndex, c.ItemID, c.ChunkID, c.AssetID,
		c.Modality, c.Score, c.Snippet)
	return err
}

func (s *Store) GetAnswer(ctx context.Context, id string) (Answer, error) {
	var a Answer
	var detail []byte
	err := s.Pool.QueryRow(ctx, `
		SELECT id, query_id, answer_text, composition_mode, model_version,
		       citation_check_passed, citation_check_detail::text, latency_ms, created_at
		FROM qa_answers WHERE id=$1`, id).
		Scan(&a.ID, &a.QueryID, &a.Answer, &a.Composition, &a.ModelVersion,
			&a.CitationCheckPass, &detail, &a.LatencyMS, &a.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return a, ErrNotFound
	}
	if err != nil {
		return a, err
	}
	_ = json.Unmarshal(detail, &a.CitationCheck)
	rows, err := s.Pool.Query(ctx, `
		SELECT c.sentence_index, c.item_id, i.sku, c.chunk_id, c.asset_id,
		       c.modality, c.score, c.snippet
		FROM qa_answer_citations c JOIN items i ON i.id=c.item_id
		WHERE c.answer_id=$1 ORDER BY c.sentence_index`, id)
	if err != nil {
		return a, err
	}
	defer rows.Close()
	for rows.Next() {
		var c CitationRow
		if err := rows.Scan(&c.SentenceIndex, &c.ItemID, &c.SKU, &c.ChunkID,
			&c.AssetID, &c.Modality, &c.Score, &c.Snippet); err != nil {
			return a, err
		}
		a.Citations = append(a.Citations, c)
	}
	return a, rows.Err()
}

func (s *Store) ListAnswers(ctx context.Context, userRef string, limit int) ([]Answer, error) {
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT a.id, a.query_id, a.answer_text, a.composition_mode, a.model_version,
		       a.citation_check_passed, a.citation_check_detail::text, a.latency_ms, a.created_at
		FROM qa_answers a JOIN qa_queries q ON q.id=a.query_id
		WHERE ($1='' OR q.user_ref=$1)
		ORDER BY a.created_at DESC LIMIT $2`, userRef, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Answer{}
	for rows.Next() {
		var a Answer
		var detail []byte
		if err := rows.Scan(&a.ID, &a.QueryID, &a.Answer, &a.Composition,
			&a.ModelVersion, &a.CitationCheckPass, &detail, &a.LatencyMS,
			&a.CreatedAt); err != nil {
			return nil, err
		}
		_ = json.Unmarshal(detail, &a.CitationCheck)
		out = append(out, a)
	}
	return out, rows.Err()
}

// ------------------------------- descriptions -------------------------------

func (s *Store) CreateSpec(ctx context.Context, sp SpecSheet) (string, error) {
	var id string
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO spec_sheets (item_id, title, category, material, dimensions,
		                         features, extra, created_by)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id`,
		sp.ItemID, sp.Title, sp.Category, sp.Material, marshal(sp.Dimensions),
		marshal(sp.Features), marshal(sp.Extra), sp.CreatedBy).Scan(&id)
	return id, err
}

func (s *Store) CreateGenerationJob(ctx context.Context, specID string, beam,
	maxLen int) (string, error) {
	var id string
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO generation_jobs (spec_id, beam_width, max_len) VALUES ($1,$2,$3)
		RETURNING id`, specID, beam, maxLen).Scan(&id)
	return id, err
}

func (s *Store) ClaimGenerationJob(ctx context.Context) (GenerationJob, error) {
	var j GenerationJob
	err := s.Pool.QueryRow(ctx, `
		UPDATE generation_jobs SET status='running', started_at=now()
		WHERE id = (SELECT id FROM generation_jobs WHERE status='queued'
		            ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
		RETURNING id, spec_id, status, beam_width, max_len, created_at, started_at`).
		Scan(&j.ID, &j.SpecID, &j.Status, &j.BeamWidth, &j.MaxLen,
			&j.CreatedAt, &j.StartedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return j, ErrNotFound
	}
	return j, err
}

func (s *Store) FinishGenerationJob(ctx context.Context, id, status string,
	model, version *string, errMsg *string) error {
	_, err := s.Pool.Exec(ctx, `
		UPDATE generation_jobs SET status=$2, model_name=$3, model_version=$4,
		       error=$5, finished_at=now()
		WHERE id=$1`, id, status, model, version, errMsg)
	return err
}

func (s *Store) InsertDrafts(ctx context.Context, jobID string, drafts []Draft) error {
	for _, d := range drafts {
		if _, err := s.Pool.Exec(ctx, `
			INSERT INTO description_drafts (job_id, rank, text, score)
			VALUES ($1,$2,$3,$4) ON CONFLICT (job_id, rank) DO UPDATE
			SET text=EXCLUDED.text, score=EXCLUDED.score`,
			jobID, d.Rank, d.Text, d.Score); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) GetSpec(ctx context.Context, id string) (SpecSheet, error) {
	var sp SpecSheet
	var dims, feats, extra []byte
	err := s.Pool.QueryRow(ctx, `
		SELECT id, item_id, title, category, material, dimensions::text,
		       features::text, extra::text, created_by, created_at
		FROM spec_sheets WHERE id=$1`, id).
		Scan(&sp.ID, &sp.ItemID, &sp.Title, &sp.Category, &sp.Material, &dims,
			&feats, &extra, &sp.CreatedBy, &sp.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return sp, ErrNotFound
	}
	_ = json.Unmarshal(dims, &sp.Dimensions)
	_ = json.Unmarshal(feats, &sp.Features)
	_ = json.Unmarshal(extra, &sp.Extra)
	return sp, err
}

func (s *Store) GetGenerationJob(ctx context.Context, id string) (GenerationJob, error) {
	var j GenerationJob
	err := s.Pool.QueryRow(ctx, `
		SELECT id, spec_id, status, model_name, model_version, beam_width, max_len,
		       error, created_at, started_at, finished_at
		FROM generation_jobs WHERE id=$1`, id).
		Scan(&j.ID, &j.SpecID, &j.Status, &j.ModelName, &j.ModelVersion,
			&j.BeamWidth, &j.MaxLen, &j.Error, &j.CreatedAt, &j.StartedAt,
			&j.FinishedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return j, ErrNotFound
	}
	if err != nil {
		return j, err
	}
	j.Spec, _ = s.getSpecOrNil(ctx, j.SpecID)
	j.Drafts, _ = s.ListDrafts(ctx, id)
	j.Reviews, _ = s.ListReviews(ctx, id)
	j.Published, _ = s.ListPublishedByJob(ctx, id)
	return j, nil
}

func (s *Store) getSpecOrNil(ctx context.Context, id string) (*SpecSheet, error) {
	sp, err := s.GetSpec(ctx, id)
	if err != nil {
		return nil, err
	}
	return &sp, nil
}

func (s *Store) ListGenerationJobs(ctx context.Context, status string, limit int) ([]GenerationJob, error) {
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT id FROM generation_jobs
		WHERE ($1='' OR status=$1) ORDER BY created_at DESC LIMIT $2`, status, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	out := []GenerationJob{}
	for _, id := range ids {
		j, err := s.GetGenerationJob(ctx, id)
		if err != nil {
			return nil, err
		}
		out = append(out, j)
	}
	return out, nil
}

func (s *Store) ListDrafts(ctx context.Context, jobID string) ([]Draft, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT id, job_id, rank, text, score FROM description_drafts
		WHERE job_id=$1 ORDER BY rank`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Draft{}
	for rows.Next() {
		var d Draft
		if err := rows.Scan(&d.ID, &d.JobID, &d.Rank, &d.Text, &d.Score); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

func (s *Store) ListReviews(ctx context.Context, jobID string) ([]Review, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT id, job_id, draft_id, reviewer, decision, edited_text, notes, created_at
		FROM editor_reviews WHERE job_id=$1 ORDER BY created_at`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Review{}
	for rows.Next() {
		var r Review
		if err := rows.Scan(&r.ID, &r.JobID, &r.DraftID, &r.Reviewer, &r.Decision,
			&r.EditedText, &r.Notes, &r.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *Store) InsertReview(ctx context.Context, r Review) error {
	_, err := s.Pool.Exec(ctx, `
		INSERT INTO editor_reviews (job_id, draft_id, reviewer, decision, edited_text, notes)
		VALUES ($1,$2,$3,$4,$5,$6)`,
		r.JobID, r.DraftID, r.Reviewer, r.Decision, r.EditedText, r.Notes)
	return err
}

// SetJobStatus transitions a description job (guard lives in workflow pkg).
func (s *Store) SetJobStatus(ctx context.Context, id, status string) error {
	tag, err := s.Pool.Exec(ctx,
		`UPDATE generation_jobs SET status=$2 WHERE id=$1`, id, status)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return nil
}

func (s *Store) CreatePublished(ctx context.Context, p Published) (string, error) {
	var id string
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO published_descriptions (item_id, job_id, draft_id, text, approved_by)
		VALUES ($1,$2,$3,$4,$5) RETURNING id`,
		p.ItemID, p.JobID, nil, p.Text, p.ApprovedBy).Scan(&id)
	return id, err
}

func (s *Store) ListPublishedByJob(ctx context.Context, jobID string) ([]Published, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT id, item_id, job_id, text, approved_by, created_at
		FROM published_descriptions WHERE job_id=$1 ORDER BY created_at`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Published{}
	for rows.Next() {
		var p Published
		if err := rows.Scan(&p.ID, &p.ItemID, &p.JobID, &p.Text, &p.ApprovedBy,
			&p.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func (s *Store) ListPublishedByItem(ctx context.Context, itemID string) ([]Published, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT id, item_id, job_id, text, approved_by, created_at
		FROM published_descriptions WHERE item_id=$1 ORDER BY created_at DESC`, itemID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Published{}
	for rows.Next() {
		var p Published
		if err := rows.Scan(&p.ID, &p.ItemID, &p.JobID, &p.Text, &p.ApprovedBy,
			&p.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// ------------------------------- platform -----------------------------------

// RoleForKeyHash implements httpapi.KeyVerifier against the api_keys table.
func (s *Store) RoleForKeyHash(hash string) (string, bool) {
	var role string
	err := s.Pool.QueryRow(context.Background(),
		`SELECT role FROM api_keys WHERE key_hash=$1 AND active`, hash).Scan(&role)
	if err != nil {
		return "", false
	}
	return role, true
}

func (s *Store) Audit(ctx context.Context, actor, action, entity, entityID string,
	detail map[string]any) error {
	_, err := s.Pool.Exec(ctx, `
		INSERT INTO audit_log (actor, action, entity, entity_id, detail)
		VALUES ($1,$2,$3,$4,$5)`, actor, action, entity, entityID, marshal(detail))
	return err
}
