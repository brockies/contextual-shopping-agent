const base = process.env.MEDUSA_BASE_URL || "http://localhost:9000";
const token = process.env.MEDUSA_SESSION_TOKEN;

if (!token) {
  console.error("MEDUSA_SESSION_TOKEN not set");
  process.exit(1);
}

const headers = {
  "Authorization": `Bearer ${token}`,
  "Content-Type": "application/json",
};

function inferSlot(title) {
  const t = title.toLowerCase();
  if (t.includes("shirt") || t.includes("t-shirt") || t.includes("tee") || t.includes("sweatshirt") || t.includes("top")) return "top";
  if (t.includes("short") || t.includes("trouser") || t.includes("pants") || t.includes("chino") || t.includes("jean") || t.includes("bottom")) return "bottom";
  if (t.includes("shoe") || t.includes("trainer") || t.includes("sneaker") || t.includes("loafer") || t.includes("boot")) return "shoes";
  if (t.includes("jacket") || t.includes("coat") || t.includes("outerwear") || t.includes("rain")) return "outerwear";
  return null;
}


async function main() {
  const res = await fetch(`${base}/admin/products?limit=100`, { headers });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();

  for (const p of data.products) {
    const slot = inferSlot(p.title);
    if (!slot) {
      if (!slot) { console.log(`Skipping (cannot infer slot): ${p.title}`); continue; }
      continue;
    }

    const nextMeta = { ...(p.metadata || {}), slot };
    const up = await fetch(`${base}/admin/products/${p.id}`, {
      method: "POST",
      headers,
      body: JSON.stringify({ metadata: nextMeta }),
    });
    if (!up.ok) throw new Error(await up.text());
    console.log(`Set slot=${slot} for ${p.title}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
