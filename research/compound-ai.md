# Sistem AI Hibrid — Arhitectură Revizuită

## Model Mental

```
Memgraph ─────────────── creierul și memoria
  ├── noduri și muchii (cunoștințe explicite și implicite)
  ├── algoritmi încorporați (PageRank, community detection, shortest path)
  ├── extensii custom (proceduri Python/C++ pentru reasoning specific)
  └── triaj inteligent → decide CE intră în contextul LLM-ului

Engram ───────────────── stratul de embedding și clustering
  ├── BIRCH: clustering semantic, decay, consolidation
  ├── embeddings Ollama: reprezentare vectorială
  └── populează noduri semantice în Memgraph

LLM local (Ollama) ───── generatorul
  └── primește context filtrat de Memgraph → produce text sau cod

Coding agent (Aider) ─── mâinile
  └── execută cod, modifică fișiere, rulează teste
```

Memgraph e centrul. Nu e un simplu storage — e engine-ul de reasoning care face triajul inteligent înainte ca orice să ajungă la LLM. Engram devine un modul care alimentează graf-ul cu cunoștințe semantice.

---

## Flow Operațional

```
Task primit
    │
    ▼
┌─────────────────────────────────────────┐
│  Memgraph: triaj                        │
│                                         │
│  1. Găsește noduri relevante            │
│  2. PageRank pe subgraf → rank importanță│
│  3. Community detection → cartierul      │
│     de cunoștințe relevant               │
│  4. Shortest path → explică legături     │
│  5. Betweenness centrality → punți       │
│     între domenii                        │
│                                         │
│  Output: context ordonat, dimensionat    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  LLM local: generare                     │
│                                         │
│  Primește context filtrat de Memgraph    │
│  Decide: răspuns NL? cod? tool call?     │
│  Produce text sau cod                    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Coding agent: execuție                  │
│                                         │
│  Aplică modificările în cod             │
│  Rulează teste, validează               │
│  Raportează rezultatul                  │
└────────────┬────────────────────────────┘
             │
             ▼
      Rezultat → store înapoi în Memgraph + Engram
```

---

## Schema Graf (Memgraph)

### Noduri

| Label | Descriere | Proprietăți |
|-------|-----------|-------------|
| `:Note` | Fișier markdown / notiță Obsidian | `title`, `path`, `vault`, `modified_at` |
| `:Concept` | Concept abstract extras din text | `name`, `description`, `embedding_id` |
| `:Code` | Entitate din cod (funcție, modul, clasă) | `name`, `file_path`, `repo`, `signature` |
| `:Cluster` | Cluster BIRCH — grupare semantică | `label`, `centroid_id`, `size`, `retention` |
| `:Tag` | Tag explicit (#tag) | `name` |
| `:Source` | Originea datelor | `name` (ex: `obsidian_vault`, `repo_x`) |

### Muchii

| Relație | De la → Către | Descriere |
|---------|---------------|-----------|
| `[:LINKS_TO]` | Note → Note | Wikilink explicit `[[A]]` din Obsidian |
| `[:TAGGED_WITH]` | Note → Tag | `#tag` extras din markdown |
| `[:BELONGS_TO]` | Note → Source | Vault-ul sau repo-ul de proveniență |
| `[:CONTAINS]` | Cluster → Note/Concept | Membri ai clusterului BIRCH |
| `[:RELATES_TO]` | Concept → Concept | Proximitate semantică (de la BIRCH) |
| `[:DEPENDS_ON]` | Code → Code | Import / dependență din cod |
| `[:IMPLEMENTED_IN]` | Concept → Code | Concept abstract codificat într-un fișier |
| `[:MENTIONED_IN]` | Concept → Note | Concept discutat într-o notiță |
| `[:BRIDGES]` | Concept → Concept | Betweenness centrality mare — punte între comunități |

### Algoritmi încorporați (Memgraph)

| Algoritm | Utilizare | Când se apelează |
|----------|-----------|------------------|
| **PageRank** | Rank importanța nodurilor în subgraf | La interogare — decide ce intră în context |
| **Community detection** (Louvain) | Identifică „cartiere" de cunoștințe | La interogare — filtrează la comunitatea relevantă |
| **Shortest path** (Dijkstra) | Explică lanțul de legături între două concepte | La interogare — răspuns la „cum se leagă X de Y" |
| **Betweenness centrality** | Identifică noduri-punte între domenii | La indexare — marchează noduri `[:BRIDGES]` |
| **Connected components** | Identifică insule de cunoștințe izolate | La mentenanță — detectează gaps în cunoștințe |

---

## FAZA 1 — Memgraph + Schema + Engram ca Modul

**Obiectiv:** Memgraph rulează, schema e creată, Engram populează noduri semantice.

### 1.1 Instalare Memgraph

```bash
docker run -p 7687:7687 -p 3000:3000 memgraph/memgraph-platform
```

### 1.2 Schema inițială (Cypher)

```cypher
-- Noduri
CREATE INDEX ON :Note(title)
CREATE INDEX ON :Concept(name)
CREATE INDEX ON :Code(name)
CREATE INDEX ON :Cluster(label)
CREATE INDEX ON :Tag(name)

-- Muchii cu proprietăți
-- (definite la ingestie, vezi mai jos)
```

### 1.3 Conector Rust → Memgraph

Modul Rust care comunică cu Memgraph via `neo4rs`:
- `insert_node(label, props)` — inserează nod
- `insert_edge(from, to, rel_type, props)` — inserează muchie
- `run_algorithm(algo_name, params)` — apelează algoritm încorporat
- `query_subgraph(node_ids)` — extrage subgraf pentru context

### 1.4 Integrare Engram → Memgraph

Când Engram face `store()`:
1. Textul se procesează normal prin BIRCH (embedding, clustering, decay)
2. **Suplimentar**: se creează un nod `:Concept` în Memgraph cu `embedding_id` legat de intrarea Engram
3. Clusterul BIRCH devine nod `:Cluster` cu muchie `[:CONTAINS]` către membri
4. Clustere apropiate primesc muchie `[:RELATES_TO]` între ele

Când Engram face `consolidate()`:
1. Intrările fuzionate devin un singur nod `:Concept` în Memgraph
2. Muchiile vechi se transferă la nodul consolidat
3. Muchia `[:CONTAINS]` se actualizează la clusterul curent

---

## FAZA 2 — Parser Obsidian + Ingestie Explicită

**Obiectiv:** Extragem relații explicite din markdown și populăm Memgraph.

### 2.1 Parser Markdown (Rust)

Folosind `pulldown-cmark` + regex:
- Extrag `[[wikilinks]]` → muchie `[:LINKS_TO]`
- Extrag `#tag-uri` → muchie `[:TAGGED_WITH]`
- Citesc frontmatter YAML → proprietăți pe nod `:Note`
- Adaug `vault_name` ca proprietate + muchie `[:BELONGS_TO]` către `:Source`

### 2.2 Parser Cod (Rust)

Folosind `tree-sitter` sau regex simplu:
- Extrag imports/dependențe → muchie `[:DEPENDS_ON]`
- Extrag definiții (funcții, clase) → nod `:Code`
- Asociez cu repo-ul → muchie `[:BELONGS_TO]` către `:Source`

### 2.3 Ingestie inițială

```bash
compound-ai ingest --obsidian ~/obsidian-vault --repo ~/project-x
```

Scanează ambele, populează Memgraph cu noduri și muchii explicite + procesează prin Engram pentru noduri semantice.

---

## FAZA 3 — MCP Server pentru Memgraph

**Obiectiv:** ExpuneMemgraph (date + algoritmi) ca MCP tools pentru LLM.

### Tools

| Tool | Parametri | Returnează |
|------|-----------|------------|
| `graph_query` | `query: String, depth: usize` | Noduri + muchii relevante, ordonate PageRank |
| `graph_path` | `from: String, to: String` | Shortest path între două concepte |
| `graph_community` | `node_id: String` | Comunitatea Louvain din care face parte nodul |
| `graph_context` | `query: String, budget: usize` | Context triat complet: noduri relevante + path-uri + comunitate, dimensionat la `budget` tokeni |
| `graph_add` | `label: String, props: Map` | Adaugă nod/muchie manual |
| `memory_search` | `query: String, repo_filter: Option<String>` | Caută în Engram (există deja), opțional filtrat pe sursă |
| `memory_store` | `content: String, source: String` | Store în Engram (există deja) |

`graph_context` e tool-ul central — Memgraph face tot triajul (PageRank, community, path) și returnează un context gata dimensionat pentru LLM.

---

## FAZA 4 — Agent Loop

**Obiectiv:** Loop-ul autonom care leagă Memgraph + LLM + Coding agent.

### 4.1 Orchestrator (Rust)

```rust
loop {
    let task = receive_task();           // de la user sau scheduler
    let context = mcp_call("graph_context", task);  // Memgraph triaj
    let response = llm_call(context, task);         // LLM generează
    match response {
        CodeChange(diff) => agent_apply(diff),      // Aider aplică
        NeedMore(query) => mcp_call("graph_context", query), // drill deeper
        Final(answer) => return answer,
    }
    // Rezultat → store înapoi
    mcp_call("memory_store", result);
    mcp_call("graph_add", new_knowledge);
}
```

### 4.2 LLM Local (Ollama)

```bash
ollama run qwen2.5-coder:7b   # sau opencoder:8b
```

Model mic, local, privat. Nu face reasoning complex — generează text/cod pe baza contextului triat de Memgraph.

### 4.3 Coding Agent (Aider)

Primen instrucțiuni de la LLM, execută în repo-uri de cod. Raportează rezultatul înapoi în loop.

---

## FAZA 5 — Bucla de Învățare (Active Learning)

**Obiectiv:** Sistemul se actualizează singur în timp real.

### 5.1 File System Watcher (`notify` crate)

Serviciu de fundal care monitorizează:
- Vault-ul Obsidian → la modificare, re-parcurge nota, actualizează Memgraph (noi `[[link-uri]]`, `#tag-uri`)
- Repo-uri de cod → la commit/push, re-parcurge fișierele modificate, actualizează dependențele

### 5.2 Actualizare Incrementală

- Obsidian: nota salvată → parser extrage diferențele → Memgraph update
- Cod: commit detectat → parser extrage diferențele → Memgraph update
- Engram: text schimbat masiv → re-embed + re-cluster + update nod `:Concept` în Memgraph

### 5.3 Învățare din Feedback

Când agent loop-ul termină un task:
- **Succes** → rezultatul se stochează în Engram + Memgraph (se creează muchie între task și soluție)
- **Eșec** → se stochează și eșecul (LLM-ul viitor va ști ce NU a funcționat)
- **Corecție umană** → se stochează ca feedback explicit, cu muchie `[:CORRECTS]` către rezultatul greșit

---

## Rezumat — Ce trebuie codezat

| Componentă | Tehnologie | ~Linii | Dependență |
|------------|-----------|--------|------------|
| **Conector Rust → Memgraph** | Rust + `neo4rs` | ~200 | Memgraph Docker |
| **Parser Obsidian/Markdown** | Rust + `pulldown-cmark` | ~150 | — |
| **Parser Cod** | Rust + `tree-sitter` / regex | ~200 | — |
| **Schema Cypher** | Cypher | ~30 | Memgraph |
| **Integrare Engram → Memgraph** | Rust (modul nou) | ~150 | Engram existent |
| **MCP Server Memgraph** | Rust (JSON-RPC stdio) | ~400 | Conector Memgraph |
| **Orchestrator Agent Loop** | Rust | ~300 | MCP + Ollama |
| **File Watcher** | Rust + `notify` | ~100 | — |
| **Extensii Memgraph** | Python | ~200 | Memgraph |

**Total: ~1730 linii noi** pe lângă Engram existent (~2700 linii).