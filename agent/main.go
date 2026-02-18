package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
)

func main() {
	godotenv.Load()

	ctx := context.Background()

	dbURL := getenv("CSA_DATABASE_URL", "postgres://csa:csa@localhost:5432/csa?sslmode=disable")
	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		log.Fatalf("db connect: %v", err)
	}
	defer pool.Close()

	http.HandleFunc("/complete-outfit", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}

		var req CompleteOutfitReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}

		resp, err := runCompleteOutfit(r.Context(), pool, req)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	// Health check
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	})

	// DB sanity check
	http.HandleFunc("/db-check", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
		defer cancel()

		var ext string
		err := pool.QueryRow(ctx,
			"SELECT extname FROM pg_extension WHERE extname='vector'").Scan(&ext)
		if err != nil {
			http.Error(w, "pgvector missing: "+err.Error(), 500)
			return
		}

		w.Write([]byte("db ok; vector ext=" + ext))
	})

	// Embed + store product
	http.HandleFunc("/embed-product", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}

		var req EmbedReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}

		embedding, err := openAIEmbed(r.Context(), req.Text)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		vec := vectorLiteral(embedding)

		_, err = pool.Exec(r.Context(), `
			INSERT INTO product_embeddings (product_id, slot, title, embedding, eco_score, price_gbp)
			VALUES ($1, $2, $3::vector, $4, $5)
			ON CONFLICT (product_id) DO UPDATE
			SET category=EXCLUDED.category,
      embedding=EXCLUDED.embedding,
      eco_score=EXCLUDED.eco_score,
      price_gbp=EXCLUDED.price_gbp
`, req.ProductID, req.Category, vec, req.EcoScore, req.PriceGBP)

		if err != nil {
			http.Error(w, "db error: "+err.Error(), 500)
			return
		}

		w.Write([]byte("ok"))
	})

	// Vector search
	http.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}

		var req SearchReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}

		if req.Limit <= 0 {
			req.Limit = 5
		}

		queryEmbedding, err := openAIEmbed(r.Context(), req.Query)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		qVec := vectorLiteral(queryEmbedding)

		rows, err := pool.Query(r.Context(), `
	SELECT product_id, eco_score, price_gbp,
	       (embedding <-> $1::vector) AS distance
	FROM product_embeddings
	WHERE embedding IS NOT NULL
	  AND ($3::int IS NULL OR eco_score >= $3)
	  AND ($4::numeric IS NULL OR price_gbp <= $4)
	ORDER BY embedding <-> $1::vector
	LIMIT $2
`, qVec, req.Limit,
			nullInt(req.MinEcoScore),
			nullNum(req.MaxPriceGBP),
		)

		if err != nil {
			http.Error(w, "query error: "+err.Error(), 500)
			return
		}
		defer rows.Close()

		var hits []Hit
		for rows.Next() {
			var h Hit
			if err := rows.Scan(&h.ProductID, &h.Title, &h.Thumbnail, &h.EcoScore, &h.PriceGBP, &h.Distance); err != nil {
				http.Error(w, err.Error(), 500)
				return
			}

			// map distance to a clearer 0-100 score (tweakable)
			score := math.Exp(-h.Distance) * 100
			h.Similarity = score

			hits = append(hits, h)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(SearchResp{Hits: hits})
	})

	http.HandleFunc("/medusa-products-count", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "GET only", 405)
			return
		}

		medusaBase := getenv("MEDUSA_BASE_URL", "http://localhost:9000")
		medusaKey := os.Getenv("MEDUSA_PUBLISHABLE_KEY")
		if medusaKey == "" {
			http.Error(w, "MEDUSA_PUBLISHABLE_KEY not set", 500)
			return
		}

		req, _ := http.NewRequestWithContext(r.Context(), "GET", medusaBase+"/store/products?limit=100", nil)
		sessionToken := os.Getenv("MEDUSA_SESSION_TOKEN")
		if sessionToken == "" {
			http.Error(w, "MEDUSA_SESSION_TOKEN not set", 500)
			return
		}
		// req.Header.Set("Authorization", "Bearer "+sessionToken)
		req.Header.Set("Authorization", "Bearer "+os.Getenv("MEDUSA_SESSION_TOKEN"))

		res, err := http.DefaultClient.Do(req)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		defer res.Body.Close()

		if res.StatusCode >= 300 {
			raw, _ := io.ReadAll(res.Body)
			http.Error(w, string(raw), 500)
			return
		}

		var payload struct {
			Products []struct {
				ID          string         `json:"id"`
				Title       string         `json:"title"`
				Thumbnail   string         `json:"thumbnail"`
				Description string         `json:"description"`
				Metadata    map[string]any `json:"metadata"`
				Categories  []struct {
					Name string `json:"name"`
				} `json:"categories"`
			} `json:"products"`
		}

		if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"count": len(payload.Products),
		})
	})

	http.HandleFunc("/index-medusa-products", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}

		medusaBase := getenv("MEDUSA_BASE_URL", "http://localhost:9000")
		medusaKey := os.Getenv("MEDUSA_PUBLISHABLE_KEY")
		if medusaKey == "" {
			http.Error(w, "MEDUSA_PUBLISHABLE_KEY not set", 500)
			return
		}

		log.Printf("INDEX: url=%s", medusaBase+"/admin/products?limit=100")
		tok := os.Getenv("MEDUSA_SESSION_TOKEN")
		log.Printf("INDEX: token_prefix=%q", func() string {
			if len(tok) > 12 {
				return tok[:12]
			}
			return tok
		}())

		req, _ := http.NewRequestWithContext(r.Context(), "GET", medusaBase+"/admin/products?limit=100", nil)
		req.Header.Set("x-publishable-api-key", medusaKey)
		req.Header.Set("Authorization", "Bearer "+tok)

		res, err := http.DefaultClient.Do(req)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		defer res.Body.Close()

		log.Printf("INDEX: medusa status=%d", res.StatusCode)

		if res.StatusCode >= 300 {
			raw, _ := io.ReadAll(res.Body)
			http.Error(w, string(raw), 500)
			return
		}

		var payload struct {
			Products []struct {
				ID          string `json:"id"`
				Title       string `json:"title"`
				Thumbnail   string `json:"thumbnail"`
				Description string `json:"description"`
				Categories  []struct {
					Name string `json:"name"`
				} `json:"categories"`
				Metadata map[string]any `json:"metadata"`
				Variants []struct {
					Prices []struct {
						Amount       int    `json:"amount"`
						CurrencyCode string `json:"currency_code"`
					} `json:"prices"`
				} `json:"variants"`
			} `json:"products"`
		}

		if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		indexed := 0
		for _, p := range payload.Products {
			category := slotFromMeta(p.Metadata)

			eco := ecoFromMeta(p.Metadata)
			price := priceFromMetaGBP(p.Metadata)

			// MVP: price not fetched yet; store 0 for now (we'll enhance later)

			card := fmt.Sprintf("TITLE: %s\nCATEGORY: %s\nDESCRIPTION: %s\nSUSTAINABILITY: eco_score=%d\nPRICE_GBP: %.2f",
				p.Title, category, p.Description, eco, price)

			emb, err := openAIEmbed(r.Context(), card)
			if err != nil {
				http.Error(w, err.Error(), 500)
				return
			}
			vec := vectorLiteral(emb)

			_, err = pool.Exec(r.Context(), `
		INSERT INTO product_embeddings (product_id, category, title, thumbnail, embedding, eco_score, price_gbp)
VALUES ($1,$2,$3,$4,$5::vector,$6,$7)
ON CONFLICT (product_id) DO UPDATE
SET category=EXCLUDED.category,
    title=EXCLUDED.title,
    thumbnail=EXCLUDED.thumbnail,
    embedding=EXCLUDED.embedding,
    eco_score=EXCLUDED.eco_score,
    price_gbp=EXCLUDED.price_gbp;
		`, p.ID, category, p.Title, p.Thumbnail, vec, eco, price)
			if err != nil {
				http.Error(w, "db upsert: "+err.Error(), 500)
				return
			}

			indexed++
		}

		w.Write([]byte(fmt.Sprintf("indexed %d products", indexed)))
	}))

	http.HandleFunc("/demo", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}

		var req CompleteOutfitReq
		_ = json.NewDecoder(r.Body).Decode(&req)

		if req.Mission == "" {
			req.Mission = "smart_casual"
		}
		if req.BudgetGBP <= 0 {
			req.BudgetGBP = 120
		}
		if req.LimitPerSlot <= 0 {
			req.LimitPerSlot = 3
		}
		if req.CartSlots == nil || len(req.CartSlots) == 0 {
			req.CartSlots = []string{"top"}
		}

		resp, err := runCompleteOutfit(r.Context(), pool, req)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))

	http.HandleFunc("/explain-outfit", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}

		var resp CompleteOutfitResp
		dec := json.NewDecoder(r.Body)
		if err := dec.Decode(&resp); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), 400)
			return
		}

		bullets, err := explainOutfitWithFallback(r.Context(), resp)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"bullets": bullets,
		})
	}))

	log.Println("Agent running on :8181")
	log.Fatal(http.ListenAndServe(":8181", nil))
}

type EmbedReq struct {
	ProductID string  `json:"product_id"`
	Category  string  `json:"category"` // top|bottom|shoes|outerwear
	Text      string  `json:"text"`
	EcoScore  int     `json:"eco_score"`
	PriceGBP  float64 `json:"price_gbp"`
}

type SearchReq struct {
	Query       string  `json:"query"`
	Limit       int     `json:"limit"`
	MaxPriceGBP float64 `json:"max_price_gbp"`
	MinEcoScore int     `json:"min_eco_score"`
}

type Hit struct {
	ProductID  string  `json:"product_id"`
	Title      string  `json:"title"`
	Thumbnail  string  `json:"thumbnail"`
	EcoScore   int     `json:"eco_score"`
	PriceGBP   float64 `json:"price_gbp"`
	Distance   float64 `json:"distance"`
	Similarity float64 `json:"similarity"`
	Reason     string  `json:"reason"`
}

type SearchResp struct {
	Hits []Hit `json:"hits"`
}

type CompleteOutfitReq struct {
	Mission      string   `json:"mission"`    // smart_casual | business_casual | outdoor_rain
	BudgetGBP    float64  `json:"budget_gbp"` // budget for add-ons
	MinEcoScore  int      `json:"min_eco_score"`
	CartSlots    []string `json:"cart_slots"`     // e.g. ["top"] or ["top","outerwear"]
	LimitPerSlot int      `json:"limit_per_slot"` // default 3
}

type SlotRecs struct {
	Slot   string `json:"slot"`
	Hits   []Hit  `json:"hits"`
	Reason string `json:"reason,omitempty"`
}

type CompleteOutfitResp struct {
	MissingSlots []string   `json:"missing_slots"`
	Results      []SlotRecs `json:"results"`
}

func openAIEmbed(ctx context.Context, text string) ([]float64, error) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY not set")
	}

	body := map[string]any{
		"model": "text-embedding-3-small",
		"input": text,
	}
	b, _ := json.Marshal(body)

	req, _ := http.NewRequestWithContext(ctx,
		"POST",
		"https://api.openai.com/v1/embeddings",
		bytes.NewReader(b))

	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode >= 300 {
		raw, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("openai error: %s", string(raw))
	}

	var parsed struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(res.Body).Decode(&parsed); err != nil {
		return nil, err
	}

	if len(parsed.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	return parsed.Data[0].Embedding, nil
}

func openAIChat(ctx context.Context, prompt string) (string, error) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return "", fmt.Errorf("OPENAI_API_KEY not set")
	}

	body := map[string]any{
		"model":       "gpt-4o-mini",
		"temperature": 0.2,
		"messages": []map[string]string{
			{"role": "system", "content": "You are a precise shopping assistant."},
			{"role": "user", "content": prompt},
		},
	}
	b, _ := json.Marshal(body)

	req, _ := http.NewRequestWithContext(ctx,
		"POST",
		"https://api.openai.com/v1/chat/completions",
		bytes.NewReader(b))

	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()

	if res.StatusCode >= 300 {
		raw, _ := io.ReadAll(res.Body)
		return "", fmt.Errorf("openai error: %s", string(raw))
	}

	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(res.Body).Decode(&parsed); err != nil {
		return "", err
	}

	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("no completion returned")
	}

	return parsed.Choices[0].Message.Content, nil
}

func vectorLiteral(v []float64) string {
	buf := bytes.NewBufferString("[")
	for i, x := range v {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(fmt.Sprintf("%g", x))
	}
	buf.WriteByte(']')
	return buf.String()
}

func getenv(k, def string) string {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	return v
}

func nullInt(v int) any {
	if v <= 0 {
		return nil
	}
	return v
}

func nullNum(v float64) any {
	if v <= 0 {
		return nil
	}
	return v
}

func requiredSlots(mission string) []string {
	switch mission {
	case "business_casual":
		return []string{"top", "bottom", "shoes"}
	case "outdoor_rain":
		return []string{"outerwear", "bottom", "shoes"}
	default: // smart_casual
		return []string{"top", "bottom", "shoes"}
	}
}

func missingSlots(required, present []string) []string {
	set := map[string]bool{}
	for _, s := range present {
		set[s] = true
	}
	var missing []string
	for _, r := range required {
		if !set[r] {
			missing = append(missing, r)
		}
	}
	return missing
}

func searchHits(ctx context.Context, pool *pgxpool.Pool, query string, limit int, maxPrice float64, minEco int, category string) ([]Hit, error) {
	qEmb, err := openAIEmbed(ctx, query)
	if err != nil {
		return nil, err
	}
	qVec := vectorLiteral(qEmb)

	rows, err := pool.Query(ctx, `
SELECT product_id, title, thumbnail, eco_score, price_gbp,
       (embedding <-> $1::vector) AS distance
FROM product_embeddings
WHERE embedding IS NOT NULL
  AND ($3::int IS NULL OR eco_score >= $3)
  AND ($4::numeric IS NULL OR price_gbp <= $4)
  AND ($5::text IS NULL OR category = $5)
ORDER BY embedding <-> $1::vector
LIMIT $2

	`, qVec, limit, nullInt(minEco), nullNum(maxPrice), nullText(category))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var hits []Hit
	for rows.Next() {
		var h Hit
		if err := rows.Scan(
			&h.ProductID,
			&h.Title,
			&h.Thumbnail,
			&h.EcoScore,
			&h.PriceGBP,
			&h.Distance,
		); err != nil {
			return nil, err
		}

		// map distance to a clearer 0-100 score (tweakable)
		score := math.Exp(-h.Distance) * 100
		h.Similarity = score

		// round distance for cleaner display
		h.Distance = math.Round(h.Distance*100) / 100

		hits = append(hits, h)
	}
	return hits, nil
}

func nullText(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func ecoFromMeta(m map[string]any) int {
	if m == nil {
		return 0
	}
	if v, ok := m["eco_score"]; ok {
		switch t := v.(type) {
		case float64:
			return int(t)
		case int:
			return t
		}
	}
	return 0
}

func normalizeCategory(name string) string {
	switch name {
	case "top", "Top":
		return "top"
	case "bottom", "Bottom":
		return "bottom"
	case "shoes", "Shoes":
		return "shoes"
	case "outerwear", "Outerwear":
		return "outerwear"
	default:
		return name
	}
}

func priceFromMetaGBP(m map[string]any) float64 {
	if m == nil {
		return 0
	}
	if v, ok := m["price_gbp"]; ok {
		switch t := v.(type) {
		case float64:
			return t
		case int:
			return float64(t)
		}
	}
	return 0
}

func slotFromMeta(m map[string]any) string {
	if m == nil {
		return ""
	}
	if v, ok := m["slot"]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func runCompleteOutfit(ctx context.Context, pool *pgxpool.Pool, req CompleteOutfitReq) (CompleteOutfitResp, error) {
	if req.LimitPerSlot <= 0 {
		req.LimitPerSlot = 3
	}

	reqSlots := requiredSlots(req.Mission)
	missing := missingSlots(reqSlots, req.CartSlots)

	perSlotBudget := req.BudgetGBP
	if len(missing) > 0 && req.BudgetGBP > 0 {
		perSlotBudget = req.BudgetGBP / float64(len(missing))
	}

	results := make([]SlotRecs, 0, len(missing))

	for _, slot := range missing {
		q := fmt.Sprintf("%s %s", req.Mission, slot)

		hits, err := searchHits(ctx, pool, q, req.LimitPerSlot, perSlotBudget, req.MinEcoScore, slot)
		if err != nil {
			return CompleteOutfitResp{}, err
		}
		if hits == nil {
			hits = []Hit{} // never return null
		}

		reason := ""
		if len(hits) == 0 {
			reason = fmt.Sprintf("No products satisfy constraints for slot=%s (slotBudget<=£%.2f, minEco=%d).",
				slot, perSlotBudget, req.MinEcoScore)
		} else {
			for i := range hits {
				hits[i].Reason = fmt.Sprintf("Matches slot=%s. Eco=%d. Price=£%.2f within slot budget £%.2f.",
					slot, hits[i].EcoScore, hits[i].PriceGBP, perSlotBudget)
			}
		}

		results = append(results, SlotRecs{Slot: slot, Hits: hits, Reason: reason})
	}

	return CompleteOutfitResp{MissingSlots: missing, Results: results}, nil
}

func explainOutfitWithFallback(ctx context.Context, resp CompleteOutfitResp) ([]string, error) {
	// Deterministic message if nothing found anywhere
	anyHits := false
	for _, r := range resp.Results {
		if len(r.Hits) > 0 {
			anyHits = true
			break
		}
	}
	if !anyHits {
		return fallbackExplain(resp), nil
	}

	bullets, explainErr := openAIExplain(ctx, resp)
	if explainErr != nil || len(bullets) == 0 {
		log.Printf("EXPLAIN: using fallback (err=%v, bullets=%d)", explainErr, len(bullets))
		return fallbackExplain(resp), nil
	}
	return bullets, nil

}

func fallbackExplain(resp CompleteOutfitResp) []string {
	out := []string{
		fmt.Sprintf("Missing slots detected: %v.", resp.MissingSlots),
		"Items were retrieved by semantic similarity for each slot, then filtered by price and eco constraints.",
	}

	for _, r := range resp.Results {
		if len(r.Hits) == 0 {
			out = append(out, fmt.Sprintf("No results for %s: %s", r.Slot, r.Reason))
			continue
		}
		h := r.Hits[0]
		out = append(out, fmt.Sprintf("Top %s pick fits constraints: Eco=%d, Price=£%.2f.", r.Slot, h.EcoScore, h.PriceGBP))
	}

	if len(out) > 5 {
		out = out[:5]
	}
	return out
}

func openAIExplain(ctx context.Context, resp CompleteOutfitResp) ([]string, error) {
	b, _ := json.Marshal(resp)

	prompt := fmt.Sprintf(`
You are a precise shopping assistant.

Given this JSON result, write 3-5 concise bullet points explaining the selection.

Rules:
- Write like a helpful shopping assistant, not a technical report.
- Avoid repeating field names (do not say “eco score for bottom”).
- Combine eco + price naturally in the same sentence.
- First bullet MUST state the missing slots exactly as provided in input_json.missing_slots.
- Mention mission, eco_score, and price/budget fit.
- If a slot has zero hits, clearly explain why using the reason field.
- Each bullet must be <= 18 words.
- Write in natural language (no "Eco score for bottom:" labels).
- Do NOT invent information.
- When referencing an item, use its title from INPUT_JSON exactly.
- Return ONLY a JSON array of strings. No extra text.
- Base every statement strictly on INPUT_JSON. Do not generalise beyond it.

INPUT_JSON:
%s
`, string(b))

	raw, err := openAIChat(ctx, prompt)

	if err != nil {
		return nil, err
	}

	log.Printf("EXPLAIN raw=%q", raw)

	bullets, err := parseBullets(raw)
	if err != nil {
		return nil, err
	}
	if len(bullets) > 5 {
		bullets = bullets[:5]
	}
	return bullets, nil
}

func parseBullets(raw string) ([]string, error) {
	s := strings.TrimSpace(raw)

	if strings.HasPrefix(s, "```") {
		// remove first line (``` or ```json)
		if i := strings.Index(s, "\n"); i >= 0 {
			s = s[i+1:]
		}
		// remove trailing ```
		if j := strings.LastIndex(s, "```"); j >= 0 {
			s = s[:j]
		}
		s = strings.TrimSpace(s)
	}

	var bullets []string
	if err := json.Unmarshal([]byte(s), &bullets); err == nil && len(bullets) > 0 {
		return bullets, nil
	}

	var wrap struct {
		Bullets []string `json:"bullets"`
	}
	if err := json.Unmarshal([]byte(s), &wrap); err == nil && len(wrap.Bullets) > 0 {
		return wrap.Bullets, nil
	}

	return nil, fmt.Errorf("invalid explain JSON: %s", raw)
}

func withCORS(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "http://localhost:5173")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, x-publishable-api-key")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		h(w, r)
	}
}
