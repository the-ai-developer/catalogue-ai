// Package httpapi: router, middleware and all /api/v1 handlers.
package httpapi

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"catalogue-ai/services/api/internal/apperr"
	"catalogue-ai/services/api/internal/config"
	"catalogue-ai/services/api/internal/modelserver"
	"catalogue-ai/services/api/internal/store"
	"catalogue-ai/services/api/internal/workflow"
)

type Server struct {
	Store *store.Store
	MS    *modelserver.Client
	Cfg   *config.Config
	Keys  KeyVerifier
}

func New(st *store.Store, ms *modelserver.Client, cfg *config.Config, keys KeyVerifier) *Server {
	return &Server{Store: st, MS: ms, Cfg: cfg, Keys: keys}
}

// ------------------------------- plumbing -----------------------------------

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, err error) {
	apiErr, ok := apperr.AsError(err)
	if !ok {
		if errors.Is(err, store.ErrNotFound) {
			apiErr = apperr.NotFoundErr("resource not found")
		} else {
			apiErr = apperr.InternalErr("internal error", err)
		}
	}
	status := http.StatusInternalServerError
	switch apiErr.Code {
	case apperr.BadRequest:
		status = http.StatusBadRequest
	case apperr.Unauthorized:
		status = http.StatusUnauthorized
	case apperr.Forbidden:
		status = http.StatusForbidden
	case apperr.NotFound:
		status = http.StatusNotFound
	case apperr.Conflict:
		status = http.StatusConflict
	case apperr.Unprocessable:
		status = http.StatusUnprocessableEntity
	case apperr.Upstream:
		status = http.StatusBadGateway
	}
	writeJSON(w, status, map[string]any{"error": map[string]any{
		"code": apiErr.Code, "message": apiErr.Message, "details": apiErr.Details}})
}

func decode(r *http.Request, dst any) error {
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(dst); err != nil {
		return apperr.BadRequestErr("invalid JSON body: " + err.Error())
	}
	return nil
}

// ------------------------------- router -------------------------------------

func (s *Server) Router() http.Handler {
	r := chi.NewRouter()
	r.Use(requestID, slogMiddleware, recoverer)
	r.Get("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, 200, map[string]any{"ok": true})
	})
	r.Get("/readyz", s.readyz)
	r.Handle("/metrics", promhttp.Handler())

	viewer := Require(s.Keys, "viewer")
	editor := Require(s.Keys, "editor")
	admin := Require(s.Keys, "admin")

	r.Route("/api/v1", func(r chi.Router) {
		r.With(admin).Post("/items", s.createItem)
		r.With(viewer).Get("/items", s.listItems)
		r.With(viewer).Get("/items/{id}", s.getItem)
		r.With(admin).Patch("/items/{id}", s.patchItem)
		r.With(admin).Delete("/items/{id}", s.deleteItem)
		r.With(admin).Post("/items/{id}/assets", s.uploadAsset)
		r.With(admin).Post("/items/{id}/ingest", s.enqueueIngest)

		r.With(viewer).Get("/jobs/ingest/{id}", s.getIngestJob)

		r.With(viewer).Post("/qa/ask", s.qaAsk)
		r.With(viewer).Get("/qa/answers/{id}", s.qaAnswer)
		r.With(viewer).Get("/qa/history", s.qaHistory)

		r.With(editor).Post("/descriptions/jobs", s.createDescriptionJob)
		r.With(viewer).Get("/descriptions/jobs", s.listDescriptionJobs)
		r.With(viewer).Get("/descriptions/jobs/{id}", s.getDescriptionJob)
		r.With(editor).Post("/descriptions/jobs/{id}/review", s.reviewDescription)
		r.With(editor).Post("/descriptions/jobs/{id}/publish", s.publishDescription)
		r.With(viewer).Get("/descriptions/published/{item_id}", s.listPublished)
	})
	return r
}

func (s *Server) readyz(w http.ResponseWriter, r *http.Request) {
	if err := s.Store.Pool.Ping(r.Context()); err != nil {
		writeJSON(w, 503, map[string]any{"ok": false, "db": err.Error()})
		return
	}
	if err := s.MS.Health(r.Context()); err != nil {
		writeJSON(w, 503, map[string]any{"ok": false, "model_server": err.Error()})
		return
	}
	writeJSON(w, 200, map[string]any{"ok": true})
}

// ------------------------------- items --------------------------------------

func (s *Server) createItem(w http.ResponseWriter, r *http.Request) {
	var in struct {
		SKU        string         `json:"sku"`
		Title      string         `json:"title"`
		Category   string         `json:"category"`
		Material   *string        `json:"material"`
		Dimensions map[string]any `json:"dimensions"`
		Features   []string       `json:"features"`
		Extra      map[string]any `json:"extra"`
		Status     string         `json:"status"`
	}
	if err := decode(r, &in); err != nil {
		writeErr(w, err)
		return
	}
	if strings.TrimSpace(in.SKU) == "" || strings.TrimSpace(in.Title) == "" ||
		strings.TrimSpace(in.Category) == "" {
		writeErr(w, apperr.BadRequestErr("sku, title and category are required"))
		return
	}
	if in.Status == "" {
		in.Status = "draft"
	}
	if in.Status != "draft" && in.Status != "active" && in.Status != "archived" {
		writeErr(w, apperr.BadRequestErr("status must be draft|active|archived"))
		return
	}
	it, err := s.Store.CreateItem(r.Context(), store.Item{
		SKU: in.SKU, Title: in.Title, Category: in.Category, Material: in.Material,
		Dimensions: in.Dimensions, Features: in.Features, Extra: in.Extra,
		Status: in.Status})
	if err != nil {
		if strings.Contains(err.Error(), "sku already exists") {
			writeErr(w, apperr.ConflictErr("sku already exists"))
			return
		}
		writeErr(w, err)
		return
	}
	_ = s.Store.Audit(r.Context(), "api", "item.create", "items", it.ID, nil)
	writeJSON(w, 201, it)
}

func (s *Server) listItems(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit, _ := strconv.Atoi(q.Get("limit"))
	items, next, err := s.Store.ListItems(r.Context(), store.ListFilter{
		Query: q.Get("query"), Category: q.Get("category"),
		Status: q.Get("status"), Limit: limit, Cursor: q.Get("cursor")})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"items": items, "next_cursor": nullable(next)})
}

func nullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func (s *Server) getItem(w http.ResponseWriter, r *http.Request) {
	it, err := s.Store.GetItem(r.Context(), chi.URLParam(r, "id"))
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, it)
}

func (s *Server) patchItem(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Title      *string        `json:"title"`
		Category   *string        `json:"category"`
		Material   *string        `json:"material"`
		Dimensions map[string]any `json:"dimensions"`
		Features   []string       `json:"features"`
		Extra      map[string]any `json:"extra"`
		Status     *string        `json:"status"`
	}
	if err := decode(r, &in); err != nil {
		writeErr(w, err)
		return
	}
	if in.Status != nil && *in.Status != "draft" && *in.Status != "active" &&
		*in.Status != "archived" {
		writeErr(w, apperr.BadRequestErr("status must be draft|active|archived"))
		return
	}
	it, err := s.Store.UpdateItem(r.Context(), chi.URLParam(r, "id"), store.ItemPatch{
		Title: in.Title, Category: in.Category, Material: in.Material,
		Dimensions: in.Dimensions, Features: in.Features, Extra: in.Extra,
		Status: in.Status})
	if err != nil {
		writeErr(w, err)
		return
	}
	_ = s.Store.Audit(r.Context(), "api", "item.update", "items", it.ID, nil)
	writeJSON(w, 200, it)
}

func (s *Server) deleteItem(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if err := s.Store.SoftDeleteItem(r.Context(), id); err != nil {
		writeErr(w, err)
		return
	}
	_ = s.Store.Audit(r.Context(), "api", "item.archive", "items", id, nil)
	w.WriteHeader(204)
}

var allowedImageMIME = map[string]string{
	"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
}

func (s *Server) uploadAsset(w http.ResponseWriter, r *http.Request) {
	itemID := chi.URLParam(r, "id")
	if _, err := s.Store.GetItem(r.Context(), itemID); err != nil {
		writeErr(w, err)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 15<<20)
	if err := r.ParseMultipartForm(15 << 20); err != nil {
		writeErr(w, apperr.BadRequestErr("multipart form: "+err.Error()))
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		writeErr(w, apperr.BadRequestErr(`missing multipart field "file"`))
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, 15<<20))
	if err != nil {
		writeErr(w, apperr.BadRequestErr("cannot read upload"))
		return
	}
	mime := header.Header.Get("Content-Type")
	ext, ok := allowedImageMIME[mime]
	if !ok {
		writeErr(w, apperr.BadRequestErr("only png/jpg/webp images are accepted"))
		return
	}
	sum := sha256.Sum256(data)
	sha := hex.EncodeToString(sum[:])
	dir := filepath.Join(s.Cfg.AssetStorageDir, sha[:2])
	if err := os.MkdirAll(dir, 0o755); err != nil {
		writeErr(w, apperr.InternalErr("storage", err))
		return
	}
	name := sha[:16] + ext
	if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
		writeErr(w, apperr.InternalErr("storage", err))
		return
	}
	var wpx, hpx *int
	if cfg, _, err := image.DecodeConfig(strings.NewReader(string(data))); err == nil {
		wpx, hpx = &cfg.Width, &cfg.Height
	}
	asset, err := s.Store.AddAsset(r.Context(), itemID, store.Asset{
		Kind: "image", URL: filepath.Join(sha[:2], name), Mime: mime,
		Width: wpx, Height: hpx, Bytes: int64(len(data)), SHA256: sha})
	if err != nil {
		writeErr(w, err)
		return
	}
	asset.URL = s.Cfg.AssetPublicBaseURL + "/" + asset.URL
	_ = s.Store.Audit(r.Context(), "api", "asset.upload", "item_assets", asset.ID, nil)
	writeJSON(w, 201, asset)
}

func (s *Server) enqueueIngest(w http.ResponseWriter, r *http.Request) {
	itemID := chi.URLParam(r, "id")
	if _, err := s.Store.GetItem(r.Context(), itemID); err != nil {
		writeErr(w, err)
		return
	}
	id, err := s.Store.CreateIngestJob(r.Context(), itemID)
	if err != nil {
		writeErr(w, err)
		return
	}
	_ = s.Store.Audit(r.Context(), "api", "ingest.enqueue", "ingest_jobs", id, nil)
	writeJSON(w, 202, map[string]any{"job_id": id, "status": "queued"})
}

func (s *Server) getIngestJob(w http.ResponseWriter, r *http.Request) {
	j, err := s.Store.GetIngestJob(r.Context(), chi.URLParam(r, "id"))
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, j)
}

// ------------------------------- QA (Project 1) ------------------------------

func (s *Server) qaAsk(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Question    string   `json:"question"`
		ItemIDs     []string `json:"item_ids"`
		TopK        int      `json:"top_k"`
		UseImages   *bool    `json:"use_images"`
		Composition string   `json:"composition"`
		UserRef     string   `json:"user_ref"`
	}
	if err := decode(r, &in); err != nil {
		writeErr(w, err)
		return
	}
	if strings.TrimSpace(in.Question) == "" {
		writeErr(w, apperr.BadRequestErr("question is required"))
		return
	}
	topK := in.TopK
	if topK <= 0 {
		topK = 6
	}
	if topK > 20 {
		writeErr(w, apperr.BadRequestErr("top_k must be <= 20"))
		return
	}
	useImages := in.UseImages == nil || *in.UseImages
	if in.Composition == "" {
		in.Composition = "extractive"
	}
	if in.Composition != "extractive" && in.Composition != "abstractive" {
		writeErr(w, apperr.BadRequestErr("composition must be extractive|abstractive"))
		return
	}
	if in.UserRef == "" {
		in.UserRef = "anonymous"
	}

	start := time.Now()
	queryID, err := s.Store.InsertQuery(r.Context(), store.Query{
		UserRef: in.UserRef, Question: in.Question, ScopeItemIDs: in.ItemIDs,
		UseImages: useImages, TopK: topK})
	if err != nil {
		writeErr(w, err)
		return
	}

	modality := "any"
	if !useImages {
		modality = "text"
	}
	search, err := s.MS.Search(r.Context(), in.Question, nil, topK, in.ItemIDs,
		modality, true)
	if err != nil {
		writeErr(w, apperr.Wrap(apperr.Upstream, "retrieval failed", err))
		return
	}

	// enrich hits with SKUs for citations
	skus := map[string]string{}
	for _, h := range search.Hits {
		if _, ok := skus[h.ItemID]; !ok {
			if it, err := s.Store.GetItem(r.Context(), h.ItemID); err == nil {
				skus[h.ItemID] = it.SKU
			}
		}
	}
	contexts := make([]modelserver.ComposeContext, 0, len(search.Hits))
	for _, h := range search.Hits {
		sku := skus[h.ItemID]
		contexts = append(contexts, modelserver.ComposeContext{
			FaissID: &h.FaissID, ItemID: h.ItemID, SKU: strPtr(sku),
			ChunkID: h.ChunkID, AssetID: h.AssetID, Modality: h.Modality,
			Score: h.Score, Text: h.Text})
	}
	composed, err := s.MS.Compose(r.Context(), in.Question, in.Composition, contexts)
	if err != nil {
		writeErr(w, apperr.Wrap(apperr.Upstream, "composition failed", err))
		return
	}

	// citations per sentence (contract §1.3)
	type apiCitation struct {
		SentenceIndex int     `json:"sentence_index"`
		ItemID        string  `json:"item_id"`
		SKU           string  `json:"sku"`
		ChunkID       *string `json:"chunk_id"`
		AssetID       *string `json:"asset_id"`
		Modality      string  `json:"modality"`
		Score         float64 `json:"score"`
		Snippet       string  `json:"snippet"`
	}
	apiCitations := []apiCitation{}
	for _, sent := range composed.Sentences {
		for _, cit := range sent.Citations {
			if cit.ContextIndex < 0 || cit.ContextIndex >= len(search.Hits) {
				continue
			}
			h := search.Hits[cit.ContextIndex]
			snippet := h.Text
			if len(snippet) > 240 {
				snippet = snippet[:240]
			}
			apiCitations = append(apiCitations, apiCitation{
				SentenceIndex: sent.Index, ItemID: h.ItemID,
				SKU: skus[h.ItemID], ChunkID: h.ChunkID, AssetID: h.AssetID,
				Modality: h.Modality, Score: cit.Score, Snippet: snippet})
			_ = s.Store.InsertCitation(r.Context(), queryID, store.CitationRow{
				SentenceIndex: sent.Index, ItemID: h.ItemID, SKU: skus[h.ItemID],
				ChunkID: h.ChunkID, AssetID: h.AssetID, Modality: h.Modality,
				Score: cit.Score, Snippet: snippet})
		}
	}

	unsourced := []int{}
	for _, d := range composed.CitationCheck.Details {
		if supported, _ := d["supported"].(bool); !supported {
			if idx, ok := d["sentence_index"].(float64); ok {
				unsourced = append(unsourced, int(idx))
			}
		}
	}
	latency := int(time.Since(start) / time.Millisecond)
	answerID, err := s.Store.InsertAnswer(r.Context(), store.Answer{
		QueryID: queryID, Answer: composed.Answer, Composition: composed.Mode,
		ModelVersion: search.Models["text"],
		CitationCheck: map[string]any{"passed": composed.CitationCheck.Passed,
			"unsourced_sentences": unsourced, "details": composed.CitationCheck.Details},
		CitationCheckPass: composed.CitationCheck.Passed, LatencyMS: latency})
	if err != nil {
		writeErr(w, err)
		return
	}
	// citation rows are keyed to the answer id for retrieval later
	_ = s.Store.Audit(r.Context(), in.UserRef, "qa.ask", "qa_answers", answerID, nil)

	writeJSON(w, 200, map[string]any{
		"answer_id":   answerID,
		"answer":      composed.Answer,
		"composition": composed.Mode,
		"citations":   apiCitations,
		"citation_check": map[string]any{
			"passed": composed.CitationCheck.Passed, "unsourced_sentences": unsourced},
		"retrieval": map[string]any{
			"text_hits":  countModality(search.Hits, "text"),
			"image_hits": countModality(search.Hits, "image"),
			"models":     search.Models},
		"latency_ms": latency,
	})
}

func countModality(hits []modelserver.Hit, modality string) int {
	n := 0
	for _, h := range hits {
		if h.Modality == modality {
			n++
		}
	}
	return n
}

func (s *Server) qaAnswer(w http.ResponseWriter, r *http.Request) {
	a, err := s.Store.GetAnswer(r.Context(), chi.URLParam(r, "id"))
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, a)
}

func (s *Server) qaHistory(w http.ResponseWriter, r *http.Request) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	answers, err := s.Store.ListAnswers(r.Context(), r.URL.Query().Get("user_ref"), limit)
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"items": answers, "next_cursor": nil})
}

// ------------------------------- descriptions (Project 2) --------------------

func (s *Server) createDescriptionJob(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Spec struct {
			ItemID     *string        `json:"item_id"`
			Title      *string        `json:"title"`
			Category   string         `json:"category"`
			Material   *string        `json:"material"`
			Dimensions map[string]any `json:"dimensions"`
			Features   []string       `json:"features"`
			Extra      map[string]any `json:"extra"`
		} `json:"spec"`
		BeamWidth int `json:"beam_width"`
		MaxLen    int `json:"max_len"`
	}
	if err := decode(r, &in); err != nil {
		writeErr(w, err)
		return
	}
	if strings.TrimSpace(in.Spec.Category) == "" {
		writeErr(w, apperr.BadRequestErr("spec.category is required"))
		return
	}
	if in.BeamWidth <= 0 {
		in.BeamWidth = 4
	}
	if in.MaxLen <= 0 {
		in.MaxLen = 192
	}
	specID, err := s.Store.CreateSpec(r.Context(), store.SpecSheet{
		ItemID: in.Spec.ItemID, Title: in.Spec.Title, Category: in.Spec.Category,
		Material: in.Spec.Material, Dimensions: in.Spec.Dimensions,
		Features: in.Spec.Features, Extra: in.Spec.Extra, CreatedBy: "editor"})
	if err != nil {
		writeErr(w, err)
		return
	}
	jobID, err := s.Store.CreateGenerationJob(r.Context(), specID, in.BeamWidth, in.MaxLen)
	if err != nil {
		writeErr(w, err)
		return
	}
	_ = s.Store.Audit(r.Context(), "editor", "description.enqueue",
		"generation_jobs", jobID, nil)
	writeJSON(w, 202, map[string]any{"job_id": jobID, "status": "queued"})
}

func (s *Server) getDescriptionJob(w http.ResponseWriter, r *http.Request) {
	j, err := s.Store.GetGenerationJob(r.Context(), chi.URLParam(r, "id"))
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, j)
}

func (s *Server) listDescriptionJobs(w http.ResponseWriter, r *http.Request) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	jobs, err := s.Store.ListGenerationJobs(r.Context(),
		r.URL.Query().Get("status"), limit)
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"items": jobs, "next_cursor": nil})
}

func (s *Server) reviewDescription(w http.ResponseWriter, r *http.Request) {
	jobID := chi.URLParam(r, "id")
	var in struct {
		Decision   string  `json:"decision"`
		DraftID    *string `json:"draft_id"`
		EditedText *string `json:"edited_text"`
		Notes      *string `json:"notes"`
	}
	if err := decode(r, &in); err != nil {
		writeErr(w, err)
		return
	}
	j, err := s.Store.GetGenerationJob(r.Context(), jobID)
	if err != nil {
		writeErr(w, err)
		return
	}
	// ---- editor gate: only draft_ready jobs, only known decisions ----
	if !workflow.CanReview(j.Status, in.Decision) {
		writeErr(w, apperr.ConflictErr(fmt.Sprintf(
			"cannot review job in status %q with decision %q", j.Status, in.Decision)))
		return
	}
	if in.Decision == workflow.DecisionApprove || in.Decision == workflow.DecisionEdit {
		if in.DraftID == nil {
			writeErr(w, apperr.BadRequestErr("draft_id is required for approve/edit"))
			return
		}
		found := false
		for _, d := range j.Drafts {
			if d.ID == *in.DraftID {
				found = true
				break
			}
		}
		if !found {
			writeErr(w, apperr.UnprocessableErr("draft_id does not belong to this job"))
			return
		}
	}
	if in.Decision == workflow.DecisionEdit &&
		(in.EditedText == nil || strings.TrimSpace(*in.EditedText) == "") {
		writeErr(w, apperr.BadRequestErr("edited_text is required for edit"))
		return
	}
	reviewer := r.Header.Get("X-Api-Key-Name")
	if reviewer == "" {
		reviewer = "editor"
	}
	if err := s.Store.InsertReview(r.Context(), store.Review{
		JobID: jobID, DraftID: in.DraftID, Reviewer: reviewer,
		Decision: in.Decision, EditedText: in.EditedText, Notes: in.Notes}); err != nil {
		writeErr(w, err)
		return
	}
	next, _ := workflow.ReviewOutcome(in.Decision)
	if err := s.Store.SetJobStatus(r.Context(), jobID, next); err != nil {
		writeErr(w, err)
		return
	}
	_ = s.Store.Audit(r.Context(), reviewer, "description."+in.Decision,
		"generation_jobs", jobID, map[string]any{"draft_id": in.DraftID})
	updated, err := s.Store.GetGenerationJob(r.Context(), jobID)
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, updated)
}

func (s *Server) publishDescription(w http.ResponseWriter, r *http.Request) {
	jobID := chi.URLParam(r, "id")
	var in struct {
		ItemID *string `json:"item_id"`
	}
	_ = decode(r, &in) // empty body is fine

	j, err := s.Store.GetGenerationJob(r.Context(), jobID)
	if err != nil {
		writeErr(w, err)
		return
	}
	// ---- editor gate: only approved text reaches the client ----
	if !workflow.CanPublish(j.Status) {
		writeErr(w, apperr.ConflictErr(fmt.Sprintf(
			"job status %q is not publishable; approval required", j.Status)))
		return
	}
	itemID := in.ItemID
	if itemID == nil && j.Spec != nil {
		itemID = j.Spec.ItemID
	}
	if itemID == nil {
		writeErr(w, apperr.UnprocessableErr(
			"publish needs item_id (job spec has no item_id)"))
		return
	}

	// publish the human-approved text: edited_text wins over the draft
	text := ""
	for i := len(j.Reviews) - 1; i >= 0; i-- {
		if j.Reviews[i].Decision == workflow.DecisionEdit &&
			j.Reviews[i].EditedText != nil {
			text = *j.Reviews[i].EditedText
			break
		}
		if j.Reviews[i].Decision == workflow.DecisionApprove &&
			j.Reviews[i].DraftID != nil {
			for _, d := range j.Drafts {
				if d.ID == *j.Reviews[i].DraftID {
					text = d.Text
					break
				}
			}
			break
		}
	}
	if text == "" && len(j.Drafts) > 0 {
		text = j.Drafts[0].Text
	}
	pubID, err := s.Store.CreatePublished(r.Context(), store.Published{
		ItemID: *itemID, JobID: jobID, Text: text, ApprovedBy: "editor"})
	if err != nil {
		writeErr(w, err)
		return
	}
	if err := s.Store.SetJobStatus(r.Context(), jobID, workflow.JobPublished); err != nil {
		writeErr(w, err)
		return
	}
	_ = s.Store.Audit(r.Context(), "editor", "description.publish",
		"published_descriptions", pubID, nil)
	writeJSON(w, 201, map[string]any{
		"id": pubID, "item_id": *itemID, "job_id": jobID,
		"text": text, "approved_by": "editor"})
}

func (s *Server) listPublished(w http.ResponseWriter, r *http.Request) {
	pubs, err := s.Store.ListPublishedByItem(r.Context(), chi.URLParam(r, "item_id"))
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"items": pubs, "next_cursor": nil})
}
