package httpapi

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	reqTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "api_requests_total", Help: "HTTP requests by route and status"},
		[]string{"route", "status"})
	reqDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name: "api_request_duration_seconds", Help: "HTTP latency by route",
		Buckets: prometheus.DefBuckets}, []string{"route"})
)

type ctxKey string

const requestIDKey ctxKey = "request_id"

func requestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			buf := make([]byte, 8)
			_, _ = rand.Read(buf)
			id = hex.EncodeToString(buf)
		}
		w.Header().Set("X-Request-ID", id)
		ctx := context.WithValue(r.Context(), requestIDKey, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func slogMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(sw, r)
		route := r.URL.Path
		reqTotal.WithLabelValues(route, http.StatusText(sw.status)).Inc()
		reqDuration.WithLabelValues(route).Observe(time.Since(start).Seconds())
		slog.Info("http", "method", r.Method, "path", route,
			"status", sw.status, "duration_ms", time.Since(start).Milliseconds(),
			"request_id", r.Context().Value(requestIDKey))
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				slog.Error("panic recovered", "err", rec, "path", r.URL.Path)
				writeJSON(w, http.StatusInternalServerError, map[string]any{
					"error": map[string]any{"code": "internal",
						"message": "internal error", "details": map[string]any{}}})
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
