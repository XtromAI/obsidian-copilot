# Semantic Search Implementation Analysis

## Overview

This document provides a comprehensive analysis of the semantic search implementation in Obsidian Copilot. The system combines vector-based semantic search with lexical (keyword-based) search to provide powerful note retrieval capabilities.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Package Dependencies](#package-dependencies)
3. [Data Storage](#data-storage)
4. [Chunking Strategy](#chunking-strategy)
5. [Embedding Models](#embedding-models)
6. [Vector Search Implementation](#vector-search-implementation)
7. [Lexical Search (v3)](#lexical-search-v3)
8. [Key Files Reference](#key-files-reference)

---

## Architecture Overview

The semantic search system has evolved through multiple versions and currently operates in two modes:

### Legacy System (Orama-based)
- **Vector Store**: Orama database with partitioned storage
- **Storage Format**: JSON files with chunked partitions
- **Embeddings**: Generated via various providers (OpenAI, Google, Cohere, etc.)
- **Search**: Hybrid mode combining vector similarity and text search

### Modern System (v3)
- **Lexical Search**: FlexSearch-based ephemeral indexes
- **Vector Store**: In-memory with optional persistence
- **Storage Format**: JSONL snapshots (mentioned in deprecation comments)
- **Search**: Tiered retrieval with grep → graph expansion → full-text search
- **Merged Approach**: Combines semantic (vector) and lexical (keyword) results

### High-Level Flow

```
User Query
    ↓
┌───────────────────────────────┐
│   Query Expansion             │
│   (Synonyms, variants, tags)  │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│   Parallel Retrieval          │
│   ├─ Semantic (Vector Search) │
│   └─ Lexical (FlexSearch)     │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│   Result Merging & Ranking    │
│   (Score normalization)       │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│   Optional Reranking          │
│   (API-based for better order)│
└───────────────────────────────┘
    ↓
    Top K Documents → LLM Context
```

---

## Package Dependencies

### Core Search Libraries

```json
{
  "@orama/orama": "^3.0.0-rc-2",        // Vector database (legacy system)
  "flexsearch": "^0.8.205",              // Full-text search engine (v3)
  "fuzzysort": "^3.1.0"                  // Fuzzy matching utilities
}
```

### LangChain Ecosystem

```json
{
  "langchain": "^1.0.1",                 // Core LangChain library
  "@langchain/core": "^1.0.1",           // Core abstractions
  "@langchain/community": "^1.0.0",      // Community integrations
  "@langchain/textsplitters": "^1.0.0"   // Text chunking utilities
}
```

### Embedding Provider Packages

```json
{
  "@langchain/openai": "^1.0.0",         // OpenAI embeddings
  "@langchain/anthropic": "^1.0.0",      // Anthropic (Claude)
  "@langchain/google-genai": "^1.0.0",   // Google Gemini
  "@langchain/cohere": "^1.0.0",         // Cohere embeddings
  "@langchain/ollama": "^1.0.0",         // Ollama (LOCAL)
  "@huggingface/inference": "^4.11.3"    // HuggingFace models
}
```

### Utilities

```json
{
  "crypto-js": "^4.1.1",                 // Hashing for document IDs
  "async-mutex": "^0.5.0",               // Concurrency control
  "p-queue": "^8.1.0"                    // Rate limiting
}
```

---

## Data Storage

### Orama-based Storage (Legacy)

**Location**: Configurable via settings
- Synced mode: `.obsidian/` (vault config directory)
- Local mode: `.copilot-index/` (vault root)

**Structure**:
```
.copilot-index/
  ├── copilot-index-chunk-{vault_hash}-metadata.json
  ├── copilot-index-chunk-{vault_hash}-0.json
  ├── copilot-index-chunk-{vault_hash}-1.json
  └── copilot-index-chunk-{vault_hash}-N.json
```

**Metadata File** (`ChunkMetadata`):
```json
{
  "numPartitions": 4,
  "vectorLength": 1536,
  "schema": {
    "id": "string",
    "title": "string",
    "path": "string",
    "content": "string",
    "embedding": "vector[1536]",
    "embeddingModel": "string",
    "created_at": "number",
    "ctime": "number",
    "mtime": "number",
    "tags": "string[]",
    "extension": "string"
  },
  "lastModified": 1234567890,
  "documentPartitions": {
    "doc_id_1": 0,
    "doc_id_2": 1
  }
}
```

**Partition Files**:
- Each partition contains a subset of documents
- Partition assignment uses djb2 hash algorithm: `Math.abs(hash(docId)) % numPartitions`
- Documents distributed across partitions for memory efficiency
- First partition includes global data + schema

**Document Structure** (`OramaDocument`):
```typescript
{
  id: string,              // MD5 hash of content
  title: string,           // File basename
  content: string,         // Chunk text with headers
  embedding: number[],     // Vector representation
  path: string,            // File path in vault
  embeddingModel: string,  // Model identifier
  created_at: number,      // Indexing timestamp
  ctime: number,           // File creation time
  mtime: number,           // File modification time
  tags: string[],          // Obsidian tags
  extension: string,       // File extension
  nchars: number,          // Character count
  metadata: object         // Frontmatter + computed fields
}
```

### v3 Storage (Modern)

**Format**: JSONL (JSON Lines) - mentioned in comments but not fully visible in codebase
**Storage**: Primarily in-memory with optional disk persistence
**Chunks**: Managed by `ChunkManager` with Map-based cache

**Chunk Structure**:
```typescript
{
  id: string,              // "note_path#chunk_index" (e.g., "note.md#0")
  notePath: string,        // Original note path
  chunkIndex: number,      // 0-based chunk position
  content: string,         // Chunk text with headers
  contentHash: string,     // Integrity validation hash
  title: string,           // Note title
  heading: string,         // Section heading
  mtime: number            // Note modification time
}
```

**Cache Management**:
- Simple Map-based cache (no LRU eviction initially)
- Memory budget: 10MB default (configurable via `lexicalSearchRamLimit`)
- Automatic invalidation on file modification
- Regeneration on cache miss

---

## Chunking Strategy

### Chunking Configuration

**Constants** (from `src/constants.ts`):
```typescript
CHUNK_SIZE = 6000  // characters per chunk
```

**Splitter**: `RecursiveCharacterTextSplitter` from LangChain
- Language: Markdown
- Chunk size: 6000 characters
- Overlap: 0 (deterministic, no overlap)
- Separators: `["\n\n", "\n", ". ", " ", ""]`

### Chunking Algorithm: Heading-First Strategy

The system uses a sophisticated "heading-first" chunking approach (implemented in `src/search/v3/chunks.ts`):

#### Step 1: Parse Document Structure
```typescript
// Extract headings using Obsidian metadata cache
const cache = app.metadataCache.getFileCache(file);
const headings = cache?.headings || [];
```

#### Step 2: Section-Based Chunking
```
Document
  ├─ Heading 1 (Section 1)
  │   └─ Content → Chunk(s)
  ├─ Heading 2 (Section 2)
  │   └─ Content → Chunk(s)
  └─ Heading N (Section N)
      └─ Content → Chunk(s)
```

**Logic**:
1. If document has no headings → treat entire content as one section
2. For each heading section:
   - Extract content from heading position to next heading (or end)
   - If section ≤ 6000 chars → single chunk
   - If section > 6000 chars → split by paragraphs using `RecursiveCharacterTextSplitter`

#### Step 3: Contextual Headers

Each chunk includes contextual information:
```markdown
NOTE TITLE: [[note_title]]

NOTE BLOCK CONTENT:

[actual chunk content]
```

This pattern (from `@langchain/textsplitters`) helps maintain context when chunks are retrieved separately.

### Chunk ID Generation

**Format**: `{note_path}#{chunk_index}`
- Example: `"Projects/AI/notes.md#0"`, `"Projects/AI/notes.md#1"`
- No padding → unlimited chunks per note
- Deterministic and reproducible

### Indexing Process

**Implementation**: `src/search/indexOperations.ts`

1. **File Filtering**:
   ```typescript
   // Respect inclusion/exclusion patterns
   const shouldIndexFile = (file, inclusions, exclusions)
   
   // Skip files:
   // - Not matching inclusions
   // - Matching exclusions
   // - Empty content
   // - Already indexed with valid embeddings
   ```

2. **Batch Processing**:
   ```typescript
   // Process in batches for rate limiting
   const batchSize = settings.embeddingBatchSize; // Default: varies by provider
   
   for (batch of allChunks) {
     // Get embeddings for batch
     const embeddings = await embeddingInstance.embedDocuments(
       batch.map(chunk => chunk.content)
     );
     
     // Save to database
     for (chunk, embedding in zip(batch, embeddings)) {
       await db.upsert({
         ...chunk.fileInfo,
         id: MD5(chunk.content),
         content: chunk.content,
         embedding: embedding,
         created_at: Date.now(),
         nchars: chunk.content.length
       });
     }
     
     // Rate limiting
     await rateLimiter.wait();
   }
   ```

3. **Checkpointing**:
   - Saves every `8 * batchSize` chunks
   - Prevents data loss on interruption
   - Allows pause/resume functionality

4. **Garbage Collection**:
   - Removes stale documents (deleted files)
   - Removes documents no longer matching filters
   - Runs before incremental indexing

---

## Embedding Models

### Supported Providers

The system supports multiple embedding providers through a unified interface (`src/LLMProviders/embeddingManager.ts`):

| Provider | Package | Local/Remote | Notes |
|----------|---------|--------------|-------|
| **Ollama** | `@langchain/ollama` | **LOCAL** | Self-hosted models |
| **LM Studio** | Custom OpenAI wrapper | **LOCAL** | OpenAI-compatible API |
| OpenAI | `@langchain/openai` | Remote | text-embedding-3-small, etc. |
| Azure OpenAI | `@langchain/openai` | Remote | Enterprise option |
| Google | `@langchain/google-genai` | Remote | Gemini embeddings |
| Cohere | `@langchain/cohere` | Remote | Multilingual support |
| Copilot Plus | Custom | Remote | Brevilabs hosted |
| OpenAI Format | Generic | Either | Custom endpoints |
| SiliconFlow | Custom | Remote | Chinese provider |

### Local Embedding Options

#### Option 1: Ollama (Recommended for Local)

**Configuration**:
```typescript
{
  provider: "ollama",
  model: "nomic-embed-text",  // or "mxbai-embed-large", "all-minilm"
  baseUrl: "http://localhost:11434",
  apiKey: "default-key"  // Not required for Ollama
}
```

**Popular Ollama Models**:
- `nomic-embed-text` (137M params, 768 dims)
- `mxbai-embed-large` (335M params, 1024 dims)
- `all-minilm` (33M params, 384 dims)

**Installation**:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull embedding model
ollama pull nomic-embed-text

# Start server (usually runs automatically)
ollama serve
```

#### Option 2: LM Studio

**Configuration**:
```typescript
{
  provider: "lmstudio",
  model: "text-embedding-ada-002",  // Model name from LM Studio
  baseUrl: "http://localhost:1234/v1",
  apiKey: "default-key"
}
```

**Setup**:
1. Download LM Studio from https://lmstudio.ai/
2. Load an embedding model (e.g., `nomic-ai/nomic-embed-text-v1.5`)
3. Start local server in LM Studio
4. Use OpenAI-compatible endpoint

### Embedding Configuration

**Base Config** (from `embeddingManager.ts`):
```typescript
{
  maxRetries: 3,
  maxConcurrency: 3,
  timeout: 10000,  // 10 seconds
  batchSize: settings.embeddingBatchSize  // Configurable
}
```

**Ollama-Specific**:
```typescript
{
  baseUrl: "http://localhost:11434",
  model: "nomic-embed-text",
  truncate: true,  // Auto-truncate long inputs
  headers: {
    Authorization: `Bearer ${apiKey}`  // Optional
  }
}
```

### Embedding API Interface

The system abstracts embedding providers through LangChain's `Embeddings` interface:

```typescript
interface Embeddings {
  embedQuery(text: string): Promise<number[]>
  embedDocuments(documents: string[]): Promise<number[][]>
}
```

**Usage Example**:
```typescript
const embeddingManager = EmbeddingsManager.getInstance();
const embeddingAPI = await embeddingManager.getEmbeddingsAPI();

// Embed single query
const queryVector = await embeddingAPI.embedQuery("What is semantic search?");
// => [0.123, -0.456, 0.789, ..., 0.321]  (768-dimensional vector)

// Embed batch of documents
const docVectors = await embeddingAPI.embedDocuments([
  "Document 1 content",
  "Document 2 content"
]);
// => [[0.1, 0.2, ...], [0.3, 0.4, ...]]
```

---

## Vector Search Implementation

### Orama Vector Search (Legacy)

**Database**: Orama v3 with built-in vector support

**Search Modes**:

1. **Vector-only**:
   ```typescript
   await search(db, {
     mode: "vector",
     vector: {
       value: queryEmbedding,
       property: "embedding"
     },
     limit: maxK,
     similarity: minSimilarityScore  // Cosine similarity threshold
   });
   ```

2. **Hybrid** (vector + text):
   ```typescript
   await search(db, {
     mode: "hybrid",
     term: "search terms",
     vector: {
       value: queryEmbedding,
       property: "embedding"
     },
     hybridWeights: {
       text: 0.4,    // TEXT_WEIGHT constant
       vector: 0.6   // 1 - TEXT_WEIGHT
     },
     limit: maxK,
     similarity: minSimilarityScore
   });
   ```

3. **Text-only** (fallback):
   ```typescript
   await search(db, {
     term: "search terms",
     properties: ["content", "title", "path"],
     limit: maxK
   });
   ```

**Similarity Metric**: Cosine similarity (built into Orama)

### HyDE (Hypothetical Document Embeddings)

**Concept**: Generate a hypothetical answer, then search for documents similar to that answer rather than the question.

**Implementation** (from `src/search/hybridRetriever.ts`):
```typescript
async rewriteQuery(query: string): Promise<string> {
  const promptTemplate = ChatPromptTemplate.fromTemplate(
    "Please write a passage to answer the question. " +
    "If you don't know the answer, just make up a passage. " +
    "\nQuestion: {question}\nPassage:"
  );
  
  const prompt = await promptTemplate.format({ question: query });
  const chatModel = await getChatModel();
  const response = await chatModel.invoke(prompt);
  
  return response.content;
}
```

**Usage**:
```typescript
// User query: "What is semantic search?"
const rewrittenQuery = await rewriteQuery(userQuery);
// => "Semantic search is a technique that..."

// Embed the hypothetical answer instead of the question
const queryVector = await embedQuery(rewrittenQuery);
```

**Benefit**: Better retrieval when query and document language differ significantly.

### Result Filtering and Merging

**Implementation**: `HybridRetriever.filterAndFormatChunks()`

1. **Explicit Chunks** (Always Included):
   - Notes explicitly mentioned with `[[note title]]` syntax
   - Retrieved directly by path
   - No similarity threshold applied

2. **Semantic Chunks** (Threshold-based):
   - Filter by cosine similarity ≥ `minSimilarityScore`
   - Default threshold: 0.1 (configurable)

3. **Deduplication**:
   - Use `Set<pageContent>` to avoid duplicate chunks
   - Maintains order: explicit chunks first, then semantic chunks

4. **Optional Reranking** (via Brevilabs API):
   ```typescript
   if (maxScore < rerankerThreshold || allScoresAreNaN) {
     const rerankResponse = await rerank(query, chunks);
     // Sort by rerank_score instead of original score
   }
   ```

---

## Lexical Search (v3)

### Architecture

The v3 system uses **ephemeral indexes** built on-demand for each search, ensuring always-fresh results without manual index management.

**Components**:

1. **GrepScanner**: Fast initial candidate discovery
2. **QueryExpander**: Generate query variants and synonyms
3. **GraphBoostCalculator**: Boost related notes via backlinks
4. **FolderBoostCalculator**: Boost notes in specific folders
5. **FullTextEngine**: FlexSearch-based full-text search
6. **ChunkManager**: Deterministic chunk generation with caching

### Search Pipeline

**Implementation**: `src/search/v3/SearchCore.ts`

```
Query: "machine learning algorithms"
  ↓
1. Query Expansion
   → variants: ["machine learning algorithms", "ML algorithms", "AI methods"]
   → salient terms: ["machine", "learning", "algorithms"]
   → tag recall: ["#ml", "#ai", "#algorithms"]
  ↓
2. Grep Scan (fast candidate discovery)
   → Search file contents for ANY query variant/term
   → Limit: 200 candidates (configurable)
   → Uses ripgrep-style content scanning
  ↓
3. Graph Expansion (optional)
   → Find notes linked to/from candidates
   → Boost scores based on link structure
  ↓
4. Chunk Generation
   → Convert candidate notes to chunks (on-demand)
   → Cache chunks for performance
  ↓
5. FlexSearch Indexing (ephemeral)
   → Create in-memory FlexSearch index
   → Index chunks with multiple fields:
     * title (weight: 3x)
     * heading (weight: 2.5x)
     * path (weight: 1.5x)
     * tags (weight: 4x)
     * body (weight: 1x)
  ↓
6. Full-Text Search
   → Search index with query variants
   → Combine scores from multiple fields
   → Apply field weights and boosts
  ↓
7. Score Normalization
   → Min-max normalization
   → Clip outliers (0.02 - 0.98)
  ↓
8. Ranking & Filtering
   → Sort by final score
   → Return top K results
```

### FlexSearch Configuration

**Engine**: FlexSearch Document index

**Tokenization**: Hybrid (ASCII + CJK bigrams)
```typescript
tokenizeMixed(str: string): string[] {
  const tokens = new Set<string>();
  const lowered = str.toLowerCase();
  
  // ASCII words (split on word boundaries)
  const asciiWords = lowered.match(/[a-z0-9]+/g) || [];
  asciiWords.forEach(word => tokens.add(word));
  
  // CJK bigrams (for Chinese, Japanese, Korean)
  const cjkChars = lowered.match(/[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+/g) || [];
  cjkChars.forEach(chars => {
    for (let i = 0; i < chars.length - 1; i++) {
      tokens.add(chars.substring(i, i + 2));
    }
  });
  
  return Array.from(tokens);
}
```

**Index Schema**:
```typescript
{
  id: "id",  // Chunk ID
  index: [
    { field: "title", weight: 3 },
    { field: "heading", weight: 2.5 },
    { field: "path", weight: 2 },
    { field: "tags", weight: 4 },
    { field: "body", weight: 1 }
  ],
  store: ["id", "notePath", "title", "heading", "chunkIndex"]
}
```

### Memory Management

**Budget**: Configurable via `lexicalSearchRamLimit` (default: 100MB)

**Tracking**:
```typescript
class MemoryManager {
  private bytesUsed: number = 0;
  private maxBytes: number;  // From settings
  
  canAddContent(contentSize: number): boolean {
    return this.bytesUsed + contentSize <= this.maxBytes;
  }
  
  addBytes(bytes: number): void {
    this.bytesUsed += bytes;
  }
}
```

**Candidate Limiting**:
- Limit: ~5 documents per MB of RAM
- Default: 500 candidates for 100MB budget

### Merged Retrieval (Semantic + Lexical)

**Implementation**: `src/search/v3/MergedSemanticRetriever.ts`

**Strategy**:
```typescript
// Run both retrievers in parallel
const [lexicalDocs, semanticDocs] = await Promise.all([
  lexicalRetriever.getRelevantDocuments(query),
  semanticRetriever.getRelevantDocuments(query)
]);

// Merge with weighted scoring
const merged = new Map<string, Document>();

for (const doc of semanticDocs) {
  const score = doc.metadata.score * SEMANTIC_WEIGHT;
  merged.set(doc.pageContent, { ...doc, metadata: { ...doc.metadata, score } });
}

for (const doc of lexicalDocs) {
  const score = doc.metadata.score * LEXICAL_WEIGHT;
  const existing = merged.get(doc.pageContent);
  
  if (existing) {
    // Combine scores for documents found by both methods
    existing.metadata.score += score;
  } else {
    merged.set(doc.pageContent, { ...doc, metadata: { ...doc.metadata, score } });
  }
}

// Sort by combined score
return Array.from(merged.values())
  .sort((a, b) => b.metadata.score - a.metadata.score)
  .slice(0, maxK);
```

---

## Key Files Reference

### Core Search Implementation

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/search/vectorStoreManager.ts` | Legacy vector store manager | `VectorStoreManager` (deprecated) |
| `src/search/dbOperations.ts` | Orama database operations | `DBOperations`, `OramaDocument` |
| `src/search/chunkedStorage.ts` | Partitioned storage for Orama | `ChunkedStorage`, `ChunkMetadata` |
| `src/search/indexOperations.ts` | Indexing pipeline | `IndexOperations`, `IndexingState` |
| `src/search/hybridRetriever.ts` | Legacy hybrid retrieval | `HybridRetriever` (LangChain) |
| `src/search/findRelevantNotes.ts` | Note discovery | `findRelevantNotes()` |

### v3 Search Implementation

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/search/v3/SearchCore.ts` | Core search orchestration | `SearchCore.retrieve()` |
| `src/search/v3/TieredLexicalRetriever.ts` | Tiered lexical retrieval | `TieredLexicalRetriever` |
| `src/search/v3/MergedSemanticRetriever.ts` | Merges semantic + lexical | `MergedSemanticRetriever` |
| `src/search/v3/chunks.ts` | Chunk management | `ChunkManager`, `Chunk` interface |
| `src/search/v3/QueryExpander.ts` | Query expansion | `QueryExpander.expand()` |
| `src/search/v3/engines/FullTextEngine.ts` | FlexSearch implementation | `FullTextEngine` |
| `src/search/v3/scanners/GrepScanner.ts` | Grep-based candidate discovery | `GrepScanner` |
| `src/search/v3/scoring/GraphBoostCalculator.ts` | Link-based scoring | `GraphBoostCalculator` |
| `src/search/v3/scoring/FolderBoostCalculator.ts` | Folder-based scoring | `FolderBoostCalculator` |
| `src/search/v3/utils/MemoryManager.ts` | Memory budget management | `MemoryManager` |
| `src/search/v3/utils/ScoreNormalizer.ts` | Score normalization | `ScoreNormalizer` |

### Embedding and LLM

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/LLMProviders/embeddingManager.ts` | Embedding provider abstraction | `EmbeddingManager` |
| `src/LLMProviders/CustomOpenAIEmbeddings.ts` | OpenAI wrapper | `CustomOpenAIEmbeddings` |
| `src/LLMProviders/CustomJinaEmbeddings.ts` | Jina embeddings | `CustomJinaEmbeddings` |

### Utilities

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/search/searchUtils.ts` | Search utilities | `shouldIndexFile()`, `getVectorLength()` |
| `src/constants.ts` | Global constants | `CHUNK_SIZE`, `TEXT_WEIGHT` |
| `src/rateLimiter.ts` | Rate limiting | `RateLimiter` |

---

## Conclusion

The Obsidian Copilot semantic search implementation is a sophisticated system that combines:

1. **Vector search** (via Orama or in-memory stores) with local or remote embeddings
2. **Lexical search** (via FlexSearch) with ephemeral indexes
3. **Intelligent chunking** with heading-first strategy
4. **Hybrid retrieval** that merges results from multiple sources
5. **Flexible embedding** support including local models (Ollama, LM Studio)

This document provides the internal architecture details needed to understand, maintain, and enhance the search system.

---

## Additional Resources

**Ollama Documentation**: https://ollama.com/
**FlexSearch Documentation**: https://github.com/nextapps-de/flexsearch
**LangChain TextSplitters**: https://js.langchain.com/docs/modules/data_connection/document_transformers/

**Obsidian Copilot Repository**: https://github.com/XtromAI/obsidian-copilot
