# Semantic Search Implementation Analysis

## Overview

This document provides a comprehensive analysis of the semantic search implementation in Obsidian Copilot. The system combines vector-based semantic search with lexical (keyword-based) search to provide powerful note retrieval capabilities for AI agents. This analysis is intended to help understand how to implement a similar feature for CLI-based AI agents, with a focus on local, API-free implementations.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Package Dependencies](#package-dependencies)
3. [Data Storage](#data-storage)
4. [Chunking Strategy](#chunking-strategy)
5. [Embedding Models](#embedding-models)
6. [Vector Search Implementation](#vector-search-implementation)
7. [Lexical Search (v3)](#lexical-search-v3)
8. [Local-Only Implementation Path](#local-only-implementation-path)
9. [Key Files Reference](#key-files-reference)

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

## Local-Only Implementation Path

### Recommended Architecture for CLI Agents

Based on the analysis, here's a recommended local-only implementation:

#### 1. Core Components

```
CLI Agent
  ├─ Embedding Service (Ollama)
  ├─ Vector Store (FAISS or Chroma)
  ├─ Lexical Search (FlexSearch or MiniSearch)
  ├─ Text Splitter (LangChain or custom)
  └─ File Watcher (for incremental updates)
```

#### 2. Technology Stack

**Embedding**:
- **Ollama** with `nomic-embed-text` model
  - Lightweight (137M params)
  - Fast inference (~50ms per embedding)
  - Good quality (768 dimensions)
  - No API costs

**Vector Store**:
- **FAISS** (Facebook AI Similarity Search)
  - Pure local (no network)
  - Fast similarity search
  - Persistent indexes
  - Python/Node.js bindings available
  
- **Alternative**: Chroma (simpler API, good for smaller datasets)

**Lexical Search**:
- **FlexSearch** (same as Obsidian Copilot v3)
  - Fast full-text search
  - Multilingual support
  - Small footprint
  - Pure JavaScript

**Text Processing**:
- **LangChain TextSplitters** or equivalent
  - Markdown-aware chunking
  - Recursive splitting
  - Deterministic results

#### 3. Implementation Steps

##### Step 1: Set Up Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull embedding model
ollama pull nomic-embed-text

# Verify
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "test"}'
```

##### Step 2: Install Dependencies

```bash
# Node.js example
npm install @langchain/ollama @langchain/textsplitters
npm install faiss-node  # or chromadb
npm install flexsearch
npm install chokidar  # File watching
```

##### Step 3: Implement Chunking

```javascript
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import fs from "fs/promises";
import crypto from "crypto";

const CHUNK_SIZE = 6000;

class DocumentChunker {
  constructor() {
    this.splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
      chunkSize: CHUNK_SIZE,
      chunkOverlap: 0,
      separators: ["\n\n", "\n", ". ", " ", ""]
    });
  }
  
  async chunkFile(filePath) {
    const content = await fs.readFile(filePath, "utf-8");
    const basename = path.basename(filePath, ".md");
    
    // Add contextual header
    const header = `\nNOTE TITLE: [[${basename}]]\n\nNOTE BLOCK CONTENT:\n\n`;
    
    const docs = await this.splitter.createDocuments([content], [], {
      chunkHeader: header,
      appendChunkOverlapHeader: false
    });
    
    return docs.map((doc, index) => ({
      id: `${filePath}#${index}`,
      path: filePath,
      chunkIndex: index,
      content: doc.pageContent,
      hash: crypto.createHash("md5").update(doc.pageContent).digest("hex")
    }));
  }
}
```

##### Step 4: Implement Embedding Service

```javascript
import { OllamaEmbeddings } from "@langchain/ollama";

class LocalEmbeddingService {
  constructor() {
    this.embeddings = new OllamaEmbeddings({
      model: "nomic-embed-text",
      baseUrl: "http://localhost:11434",
      truncate: true
    });
  }
  
  async embedQuery(text) {
    return await this.embeddings.embedQuery(text);
  }
  
  async embedDocuments(texts) {
    return await this.embeddings.embedDocuments(texts);
  }
}
```

##### Step 5: Implement Vector Store

```javascript
import faiss from "faiss-node";
import fs from "fs/promises";

class LocalVectorStore {
  constructor(dimension = 768) {
    this.dimension = dimension;
    this.index = new faiss.IndexFlatL2(dimension);
    this.documents = new Map();  // id -> document
    this.idToIdx = new Map();    // id -> FAISS index
    this.idxToId = new Map();    // FAISS index -> id
  }
  
  async addDocuments(documents, embeddings) {
    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      const embedding = embeddings[i];
      
      const idx = this.index.ntotal();
      this.index.add(embedding);
      
      this.documents.set(doc.id, doc);
      this.idToIdx.set(doc.id, idx);
      this.idxToId.set(idx, doc.id);
    }
  }
  
  async search(queryEmbedding, k = 10, threshold = 0.1) {
    const results = this.index.search(queryEmbedding, k);
    
    return results.labels.map((idx, i) => {
      const docId = this.idxToId.get(idx);
      const doc = this.documents.get(docId);
      const distance = results.distances[i];
      
      // Convert L2 distance to similarity score (0-1)
      const similarity = 1 / (1 + distance);
      
      return {
        document: doc,
        score: similarity
      };
    }).filter(result => result.score >= threshold);
  }
  
  async save(filepath) {
    await faiss.write_index(this.index, filepath + ".faiss");
    await fs.writeFile(
      filepath + ".meta.json",
      JSON.stringify({
        documents: Array.from(this.documents.entries()),
        idToIdx: Array.from(this.idToIdx.entries()),
        idxToId: Array.from(this.idxToId.entries())
      })
    );
  }
  
  async load(filepath) {
    this.index = await faiss.read_index(filepath + ".faiss");
    const meta = JSON.parse(await fs.readFile(filepath + ".meta.json", "utf-8"));
    
    this.documents = new Map(meta.documents);
    this.idToIdx = new Map(meta.idToIdx);
    this.idxToId = new Map(meta.idxToId);
  }
}
```

##### Step 6: Implement Lexical Search

```javascript
import FlexSearch from "flexsearch";

class LocalLexicalSearch {
  constructor() {
    this.index = new FlexSearch.Document({
      encode: false,
      tokenize: this.tokenizeMixed.bind(this),
      document: {
        id: "id",
        index: [
          { field: "title", weight: 3 },
          { field: "path", weight: 2 },
          { field: "content", weight: 1 }
        ]
      }
    });
  }
  
  tokenizeMixed(str) {
    const tokens = new Set();
    const lowered = str.toLowerCase();
    
    // ASCII words
    const words = lowered.match(/[a-z0-9]+/g) || [];
    words.forEach(word => tokens.add(word));
    
    // CJK bigrams (if needed)
    const cjk = lowered.match(/[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+/g) || [];
    cjk.forEach(chars => {
      for (let i = 0; i < chars.length - 1; i++) {
        tokens.add(chars.substring(i, i + 2));
      }
    });
    
    return Array.from(tokens);
  }
  
  addDocuments(documents) {
    documents.forEach(doc => {
      this.index.add({
        id: doc.id,
        title: doc.title || path.basename(doc.path, ".md"),
        path: doc.path,
        content: doc.content
      });
    });
  }
  
  search(query, limit = 10) {
    return this.index.search(query, limit);
  }
}
```

##### Step 7: Implement Hybrid Search

```javascript
class HybridSearch {
  constructor(vectorStore, lexicalSearch, embeddingService) {
    this.vectorStore = vectorStore;
    this.lexicalSearch = lexicalSearch;
    this.embeddingService = embeddingService;
  }
  
  async search(query, options = {}) {
    const {
      maxK = 10,
      vectorWeight = 0.6,
      lexicalWeight = 0.4,
      minSimilarity = 0.1
    } = options;
    
    // Run both searches in parallel
    const [vectorResults, lexicalResults] = await Promise.all([
      this.searchVector(query, maxK * 2, minSimilarity),
      this.searchLexical(query, maxK * 2)
    ]);
    
    // Merge results with weighted scoring
    const merged = new Map();
    
    for (const result of vectorResults) {
      const score = result.score * vectorWeight;
      merged.set(result.document.id, {
        document: result.document,
        score: score,
        sources: ["vector"]
      });
    }
    
    for (const result of lexicalResults) {
      const id = result.id;
      const score = result.score * lexicalWeight;
      
      if (merged.has(id)) {
        merged.get(id).score += score;
        merged.get(id).sources.push("lexical");
      } else {
        merged.set(id, {
          document: result.document,
          score: score,
          sources: ["lexical"]
        });
      }
    }
    
    // Sort and return top K
    return Array.from(merged.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, maxK);
  }
  
  async searchVector(query, k, threshold) {
    const queryEmbedding = await this.embeddingService.embedQuery(query);
    return await this.vectorStore.search(queryEmbedding, k, threshold);
  }
  
  async searchLexical(query, k) {
    const results = this.lexicalSearch.search(query, k);
    // FlexSearch returns array of result arrays by field
    // Flatten and deduplicate
    const seen = new Set();
    const flattened = [];
    
    results.forEach(fieldResults => {
      if (Array.isArray(fieldResults)) {
        fieldResults.forEach(result => {
          if (!seen.has(result.id)) {
            seen.add(result.id);
            flattened.push(result);
          }
        });
      }
    });
    
    return flattened;
  }
}
```

##### Step 8: Implement Indexing Pipeline

```javascript
import chokidar from "chokidar";
import path from "path";

class IndexingPipeline {
  constructor(
    documentsPath,
    chunker,
    embeddingService,
    vectorStore,
    lexicalSearch
  ) {
    this.documentsPath = documentsPath;
    this.chunker = chunker;
    this.embeddingService = embeddingService;
    this.vectorStore = vectorStore;
    this.lexicalSearch = lexicalSearch;
    this.watcher = null;
  }
  
  async indexAll() {
    console.log("Starting full index...");
    
    // Find all markdown files
    const files = await this.findMarkdownFiles(this.documentsPath);
    console.log(`Found ${files.length} files`);
    
    // Process in batches
    const batchSize = 10;
    for (let i = 0; i < files.length; i += batchSize) {
      const batch = files.slice(i, i + batchSize);
      await this.indexBatch(batch);
      console.log(`Indexed ${Math.min(i + batchSize, files.length)}/${files.length} files`);
    }
    
    // Save indexes
    await this.vectorStore.save(path.join(this.documentsPath, ".index", "vector"));
    console.log("Indexing complete!");
  }
  
  async indexBatch(files) {
    // Chunk all files
    const allChunks = [];
    for (const file of files) {
      const chunks = await this.chunker.chunkFile(file);
      allChunks.push(...chunks);
    }
    
    if (allChunks.length === 0) return;
    
    // Generate embeddings
    const contents = allChunks.map(c => c.content);
    const embeddings = await this.embeddingService.embedDocuments(contents);
    
    // Add to vector store
    await this.vectorStore.addDocuments(allChunks, embeddings);
    
    // Add to lexical search
    this.lexicalSearch.addDocuments(allChunks);
  }
  
  async findMarkdownFiles(dir) {
    const { globby } = await import("globby");
    return await globby("**/*.md", { cwd: dir, absolute: true });
  }
  
  watchForChanges() {
    this.watcher = chokidar.watch("**/*.md", {
      cwd: this.documentsPath,
      ignoreInitial: true
    });
    
    this.watcher.on("add", async (filepath) => {
      console.log(`File added: ${filepath}`);
      await this.indexBatch([path.join(this.documentsPath, filepath)]);
    });
    
    this.watcher.on("change", async (filepath) => {
      console.log(`File changed: ${filepath}`);
      // Remove old chunks and reindex
      await this.removeFileChunks(filepath);
      await this.indexBatch([path.join(this.documentsPath, filepath)]);
    });
    
    this.watcher.on("unlink", async (filepath) => {
      console.log(`File removed: ${filepath}`);
      await this.removeFileChunks(filepath);
    });
  }
  
  async removeFileChunks(filepath) {
    // Implementation depends on vector store API
    // For FAISS, you'd need to rebuild the index
    // For Chroma, you can delete by metadata
  }
  
  stopWatching() {
    if (this.watcher) {
      this.watcher.close();
    }
  }
}
```

##### Step 9: Put It All Together

```javascript
// main.js
import { DocumentChunker } from "./chunker.js";
import { LocalEmbeddingService } from "./embeddings.js";
import { LocalVectorStore } from "./vector-store.js";
import { LocalLexicalSearch } from "./lexical-search.js";
import { HybridSearch } from "./hybrid-search.js";
import { IndexingPipeline } from "./indexing.js";

async function main() {
  const documentsPath = "./documents";
  
  // Initialize components
  const chunker = new DocumentChunker();
  const embeddingService = new LocalEmbeddingService();
  const vectorStore = new LocalVectorStore(768); // nomic-embed-text dimension
  const lexicalSearch = new LocalLexicalSearch();
  
  // Check if index exists
  const indexPath = path.join(documentsPath, ".index", "vector");
  const indexExists = await fs.access(indexPath + ".faiss")
    .then(() => true)
    .catch(() => false);
  
  if (indexExists) {
    // Load existing index
    console.log("Loading existing index...");
    await vectorStore.load(indexPath);
  } else {
    // Build new index
    const pipeline = new IndexingPipeline(
      documentsPath,
      chunker,
      embeddingService,
      vectorStore,
      lexicalSearch
    );
    
    await pipeline.indexAll();
    
    // Watch for changes (optional)
    pipeline.watchForChanges();
  }
  
  // Initialize search
  const search = new HybridSearch(vectorStore, lexicalSearch, embeddingService);
  
  // Example search
  const results = await search.search("machine learning algorithms", {
    maxK: 10,
    vectorWeight: 0.6,
    lexicalWeight: 0.4,
    minSimilarity: 0.1
  });
  
  console.log("Search results:");
  results.forEach((result, i) => {
    console.log(`${i + 1}. [${result.score.toFixed(3)}] ${result.document.path}`);
    console.log(`   Sources: ${result.sources.join(", ")}`);
    console.log(`   Preview: ${result.document.content.substring(0, 100)}...`);
  });
}

main().catch(console.error);
```

#### 4. Performance Considerations

**Embedding Speed** (Ollama on M1 Mac):
- nomic-embed-text: ~50ms per embedding
- Batch of 10: ~200ms
- Throughput: ~50 embeddings/second

**Index Size Estimation**:
- 1000 notes × 6 chunks/note = 6000 chunks
- 768-dimensional vectors × 4 bytes = 3KB per vector
- Total: ~18MB for vectors
- FAISS index overhead: ~20-25MB total

**Search Performance**:
- FAISS search: <10ms for 6000 vectors
- FlexSearch: <5ms for full-text
- Embedding query: ~50ms
- Total: ~65ms per search

#### 5. Advantages of Local-Only Approach

1. **No API Costs**: Zero recurring expenses
2. **Privacy**: All data stays local
3. **Offline**: Works without internet
4. **Speed**: No network latency
5. **Scalability**: Handles thousands of documents easily
6. **Control**: Full control over models and configuration

#### 6. Limitations and Mitigations

**Limitation 1: Embedding Quality**
- Local models may be less accurate than GPT-4 embeddings
- **Mitigation**: Use `nomic-embed-text` or `mxbai-embed-large` (competitive with commercial models)

**Limitation 2: Initial Indexing Time**
- Embedding 1000 documents takes ~20 minutes
- **Mitigation**: Incremental updates, background processing, progress indicators

**Limitation 3: Hardware Requirements**
- Requires ~4GB RAM for Ollama + embeddings
- **Mitigation**: Use smaller models (`all-minilm` requires <1GB)

**Limitation 4: Vector Store Updates**
- FAISS doesn't support efficient deletion
- **Mitigation**: Periodic index rebuilds or use Chroma instead

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

For a CLI-based AI agent targeting local-only operation:

- Use **Ollama** with `nomic-embed-text` for embeddings
- Use **FAISS** or **Chroma** for vector storage
- Use **FlexSearch** for lexical search
- Implement **incremental indexing** with file watching
- Follow the **chunking strategy** from this codebase (heading-first, 6000 chars)

The provided implementation examples should serve as a solid foundation for building a local semantic search system that doesn't rely on external APIs while maintaining good performance and search quality.

---

## Additional Resources

**Ollama Documentation**: https://ollama.com/
**FAISS Documentation**: https://faiss.ai/
**FlexSearch Documentation**: https://github.com/nextapps-de/flexsearch
**LangChain TextSplitters**: https://js.langchain.com/docs/modules/data_connection/document_transformers/

**Obsidian Copilot Repository**: https://github.com/XtromAI/obsidian-copilot
