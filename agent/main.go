package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

func main() {
	ctx := context.Background()

	dbURL := getenv("DATABASE_URL", "postgres://csa:csa@localhost:5432/csa?sslmode=disable")
	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		log.Fatalf("db connect: %v", err)
	}
	defer pool.Close()

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	})

	http.HandleFunc("/db-check", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
		defer cancel()

		// Check pgvector extension exists
		var ext string
		if err := pool.QueryRow(ctx, "SELECT extname FROM pg_extension WHERE extname='vector'").Scan(&ext); err != nil {
			http.Error(w, "pgvector extension not found: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// Simple insert without embedding yet (NULL embedding is OK)
		_, err := pool.Exec(ctx, `
			INSERT INTO product_embeddings (product_id, embedding, eco_score, price_gbp)
			VALUES ($1, NULL, $2, $3)
			ON CONFLICT (product_id) DO UPDATE SET eco_score=EXCLUDED.eco_score, price_gbp=EXCLUDED.price_gbp
		`, "sanity-prod", 80, 49.99)
		if err != nil {
			http.Error(w, "insert failed: "+err.Error(), http.StatusInternalServerError)
			return
		}

		var eco int
		if err := pool.QueryRow(ctx, "SELECT eco_score FROM product_embeddings WHERE product_id=$1", "sanity-prod").Scan(&eco); err != nil {
			http.Error(w, "select failed: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Write([]byte("db ok; vector ext=" + ext + "; eco_score=" + itoa(eco)))
	})

	log.Println("Agent running on :8181")
	log.Fatal(http.ListenAndServe(":8181", nil))
}

func getenv(k, def string) string {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	return v
}

func itoa(i int) string {
	// tiny local helper to avoid extra imports
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var b [20]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + (i % 10))
		i /= 10
	}
	if neg {
		pos--
		b[pos] = '-'
	}
	return string(b[pos:])
}
