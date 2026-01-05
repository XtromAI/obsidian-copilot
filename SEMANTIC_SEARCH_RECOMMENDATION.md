# Semantic Search: Simplified Modern Implementation

## Overview

This document provides recommendations for implementing a simplified, modern semantic search system that prioritizes:
- **Local-only operation** (no external API dependencies)
- **CLI agent accessibility** (via standardized protocols)
- **Minimal dependencies** (only what's essential within the plugin)
- **Simple deployment** (easy for users to set up and maintain)

## Design Philosophy

### Core Principles

1. **Everything Runs Locally**: No cloud APIs, no external services
2. **Plugin-Embedded**: All search logic lives within the Obsidian plugin
3. **Zero Configuration**: Works out-of-the-box with sensible defaults
4. **CLI-First**: Designed for programmatic access by external tools

## Recommended Architecture

### High-Level Design

```
┌─────────────────────────────────────────┐
│         CLI Agent (Your Tool)           │
└─────────────────┬───────────────────────┘
                  │ IPC/JSON-RPC
                  ↓
┌─────────────────────────────────────────┐
│      Obsidian Plugin (This Repo)        │
│  ┌───────────────────────────────────┐  │
│  │  Search API Layer                 │  │
│  └───────────┬───────────────────────┘  │
│              │                           │
│  ┌───────────▼───────────────────────┐  │
│  │  Unified Search Engine            │  │
│  │  - Lexical (BM25)                 │  │
│  │  - Semantic (Local Embeddings)    │  │
│  └───────────┬───────────────────────┘  │
│              │                           │
│  ┌───────────▼───────────────────────┐  │
│  │  Storage Layer                    │  │
│  │  - In-memory index                │  │
│  │  - Optional disk cache            │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Implementation Details

### 1. Local Embedding Model

**Recommendation**: Use WebAssembly-based embeddings that run entirely in the plugin process.

**Why Not Ollama/LM Studio?**
- Requires separate installation and management
- Adds deployment complexity
- Creates external dependency

**Recommended: Transformers.js**

```typescript
// Use Transformers.js for in-process embeddings
import { pipeline } from '@xenova/transformers';

class LocalEmbedding {
  private embedder: any;
  
  async initialize() {
    // Load small, efficient model (runs in WebAssembly)
    this.embedder = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2',  // 23MB model, 384 dimensions
      { device: 'wasm' }
    );
  }
  
  async embed(text: string): Promise<number[]> {
    const output = await this.embedder(text, {
      pooling: 'mean',
      normalize: true
    });
    return Array.from(output.data);
  }
}
```

**Benefits**:
- No external process required
- Works offline immediately
- ~23MB model size (acceptable for plugin)
- Fast inference (~50-100ms per query)
- Cross-platform (Windows, Mac, Linux)

### 2. Vector Storage

**Recommendation**: In-memory HNSW index with optional persistence.

**Why Not Orama/FAISS?**
- Orama adds significant bundle size
- FAISS requires native bindings
- Both add complexity

**Recommended: hnswlib-wasm**

```typescript
import { HierarchicalNSW } from 'hnswlib-wasm';

class VectorIndex {
  private index: HierarchicalNSW;
  private documents: Map<number, string>;
  
  constructor(dimensions: number = 384) {
    this.index = new HierarchicalNSW('cosine', dimensions);
    this.documents = new Map();
  }
  
  async add(id: number, vector: number[], content: string) {
    this.index.addPoint(vector, id);
    this.documents.set(id, content);
  }
  
  search(queryVector: number[], k: number = 10): Array<{id: number, distance: number}> {
    return this.index.searchKnn(queryVector, k);
  }
  
  // Optional: persist to disk
  async save(path: string) {
    const data = {
      index: this.index.serialize(),
      documents: Array.from(this.documents.entries())
    };
    await fs.writeFile(path, JSON.stringify(data));
  }
}
```

**Benefits**:
- Pure WebAssembly (no native dependencies)
- Fast k-NN search (sub-millisecond for 10k vectors)
- Small bundle size (~200KB)
- Memory efficient

### 3. Lexical Search

**Recommendation**: Simple BM25 implementation without external libraries.

```typescript
class BM25Search {
  private documents: Array<{id: string, tokens: string[], content: string}> = [];
  private idf: Map<string, number> = new Map();
  private avgDocLength: number = 0;
  
  // Simple tokenizer
  private tokenize(text: string): string[] {
    return text.toLowerCase()
      .split(/[\s.,!?;:()\[\]{}'"]+/)
      .filter(t => t.length > 2);
  }
  
  add(id: string, content: string) {
    const tokens = this.tokenize(content);
    this.documents.push({ id, tokens, content });
    this.updateIDF();
  }
  
  search(query: string, k: number = 10): Array<{id: string, score: number}> {
    const queryTokens = this.tokenize(query);
    const scores = this.documents.map(doc => ({
      id: doc.id,
      score: this.calculateBM25(queryTokens, doc.tokens)
    }));
    
    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }
  
  private calculateBM25(queryTokens: string[], docTokens: string[]): number {
    // Standard BM25 implementation
    const k1 = 1.5, b = 0.75;
    let score = 0;
    
    for (const term of queryTokens) {
      const tf = docTokens.filter(t => t === term).length;
      const idf = this.idf.get(term) || 0;
      const docLen = docTokens.length;
      
      score += idf * (tf * (k1 + 1)) / 
        (tf + k1 * (1 - b + b * (docLen / this.avgDocLength)));
    }
    
    return score;
  }
  
  private updateIDF() {
    // Calculate IDF scores
    const N = this.documents.length;
    const termDocCount = new Map<string, number>();
    
    for (const doc of this.documents) {
      const uniqueTerms = new Set(doc.tokens);
      uniqueTerms.forEach(term => {
        termDocCount.set(term, (termDocCount.get(term) || 0) + 1);
      });
    }
    
    termDocCount.forEach((docCount, term) => {
      this.idf.set(term, Math.log((N - docCount + 0.5) / (docCount + 0.5) + 1));
    });
    
    this.avgDocLength = this.documents.reduce((sum, doc) => 
      sum + doc.tokens.length, 0) / N;
  }
}
```

**Benefits**:
- Zero dependencies
- Simple to understand and maintain
- Proven algorithm
- Fast for moderate collections (<100k docs)

### 4. Chunking Strategy

**Recommendation**: Simple fixed-size chunking with overlap.

```typescript
class SimpleChunker {
  private chunkSize: number = 512;  // tokens
  private overlap: number = 50;      // tokens
  
  chunk(text: string, documentId: string): Array<{id: string, content: string}> {
    const words = text.split(/\s+/);
    const chunks: Array<{id: string, content: string}> = [];
    
    for (let i = 0; i < words.length; i += this.chunkSize - this.overlap) {
      const chunkWords = words.slice(i, i + this.chunkSize);
      const chunkId = `${documentId}#${chunks.length}`;
      const content = chunkWords.join(' ');
      
      chunks.push({ id: chunkId, content });
      
      if (i + this.chunkSize >= words.length) break;
    }
    
    return chunks;
  }
}
```

**Benefits**:
- Predictable behavior
- No dependency on LangChain
- Easy to debug
- Works well with small embedding models

### 5. CLI Interface

**Recommendation**: JSON-RPC over stdio for CLI agent communication.

```typescript
// Plugin exposes JSON-RPC server
class SearchRPCServer {
  private searchEngine: UnifiedSearch;
  
  constructor() {
    this.searchEngine = new UnifiedSearch();
    this.setupRPCHandler();
  }
  
  private setupRPCHandler() {
    // Listen on stdin for JSON-RPC requests
    process.stdin.on('data', async (data) => {
      const request = JSON.parse(data.toString());
      const response = await this.handleRequest(request);
      process.stdout.write(JSON.stringify(response) + '\n');
    });
  }
  
  private async handleRequest(request: any) {
    switch (request.method) {
      case 'search':
        return {
          id: request.id,
          result: await this.searchEngine.search(
            request.params.query,
            request.params.options
          )
        };
      
      case 'index':
        return {
          id: request.id,
          result: await this.searchEngine.indexDocument(
            request.params.path,
            request.params.content
          )
        };
      
      default:
        return {
          id: request.id,
          error: { code: -32601, message: 'Method not found' }
        };
    }
  }
}
```

**CLI Agent Example**:

```typescript
// CLI tool connects via stdio
class ObsidianSearchClient {
  private child: ChildProcess;
  private requestId: number = 0;
  
  async connect() {
    this.child = spawn('obsidian', ['--search-rpc']);
  }
  
  async search(query: string, options?: any): Promise<SearchResult[]> {
    const request = {
      jsonrpc: '2.0',
      method: 'search',
      params: { query, options },
      id: ++this.requestId
    };
    
    this.child.stdin.write(JSON.stringify(request) + '\n');
    
    return new Promise((resolve) => {
      this.child.stdout.once('data', (data) => {
        const response = JSON.parse(data.toString());
        resolve(response.result);
      });
    });
  }
}
```

**Benefits**:
- Standard protocol (JSON-RPC 2.0)
- No HTTP server needed
- Works with any language
- Secure (no network exposure)

### 6. Unified Search Engine

**Recommendation**: Combine lexical and semantic search with simple fusion.

```typescript
class UnifiedSearch {
  private lexical: BM25Search;
  private semantic: VectorIndex;
  private embedder: LocalEmbedding;
  private chunker: SimpleChunker;
  
  constructor() {
    this.lexical = new BM25Search();
    this.semantic = new VectorIndex(384);
    this.embedder = new LocalEmbedding();
    this.chunker = new SimpleChunker();
  }
  
  async indexDocument(path: string, content: string) {
    // Chunk the document
    const chunks = this.chunker.chunk(content, path);
    
    // Index in both systems
    for (const chunk of chunks) {
      // Lexical index
      this.lexical.add(chunk.id, chunk.content);
      
      // Semantic index
      const embedding = await this.embedder.embed(chunk.content);
      const numId = this.pathToNumId(chunk.id);
      await this.semantic.add(numId, embedding, chunk.content);
    }
  }
  
  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    const k = options.maxResults || 10;
    
    // Get results from both indexes
    const lexicalResults = this.lexical.search(query, k * 2);
    
    const queryEmbedding = await this.embedder.embed(query);
    const semanticResults = this.semantic.search(queryEmbedding, k * 2);
    
    // Reciprocal Rank Fusion (simple and effective)
    const fusedScores = new Map<string, number>();
    const k_rrf = 60;  // RRF constant
    
    lexicalResults.forEach((result, rank) => {
      const score = 1 / (k_rrf + rank + 1);
      fusedScores.set(result.id, (fusedScores.get(result.id) || 0) + score);
    });
    
    semanticResults.forEach((result, rank) => {
      const id = this.numIdToPath(result.id);
      const score = 1 / (k_rrf + rank + 1);
      fusedScores.set(id, (fusedScores.get(id) || 0) + score);
    });
    
    // Sort by fused score and return top k
    return Array.from(fusedScores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, k)
      .map(([id, score]) => ({
        id,
        score,
        content: this.getContent(id)
      }));
  }
  
  private pathToNumId(path: string): number {
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < path.length; i++) {
      hash = ((hash << 5) - hash) + path.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash);
  }
  
  private numIdToPath(id: number): string {
    // Maintain reverse mapping
    return this.semantic.documents.get(id) || '';
  }
  
  private getContent(id: string): string {
    // Retrieve content from documents
    return this.semantic.documents.get(this.pathToNumId(id)) || '';
  }
}
```

**Benefits**:
- Simple reciprocal rank fusion (proven effective)
- No complex merging logic
- Both lexical and semantic benefits
- Easy to understand and debug

## Package Dependencies

### Required (Minimal Set)

```json
{
  "@xenova/transformers": "^2.17.0",    // WebAssembly embeddings (~500KB)
  "hnswlib-wasm": "^1.0.0"              // Vector search (~200KB)
}
```

**Total Bundle Impact**: ~700KB (acceptable for modern plugin)

### Not Required

```json
{
  // ❌ No LangChain (too heavy, not needed)
  // ❌ No @orama/orama (adds complexity)
  // ❌ No FlexSearch (can implement BM25 ourselves)
  // ❌ No external LLM provider packages
  // ❌ No Ollama/LM Studio dependencies
}
```

## Implementation Roadmap

### Phase 1: Core Search (Week 1)

1. Implement BM25 lexical search (no dependencies)
2. Integrate Transformers.js for embeddings
3. Add hnswlib-wasm for vector search
4. Create simple chunking logic

### Phase 2: Integration (Week 2)

1. Add JSON-RPC interface for CLI agents
2. Implement unified search with RRF fusion
3. Add document indexing API
4. Create persistence layer (optional)

### Phase 3: Optimization (Week 3)

1. Add incremental indexing
2. Optimize memory usage
3. Add search result caching
4. Performance testing and tuning

## Usage Examples

### For Plugin Users

**Zero Configuration**: Search works immediately after installation.

```typescript
// Plugin automatically indexes vault on startup
const plugin = new SemanticSearchPlugin();
await plugin.onload();  // Indexes vault in background

// Search is immediately available
const results = await plugin.search("machine learning algorithms");
```

### For CLI Agent Developers

**Simple Integration**: Standard JSON-RPC protocol.

```python
# Python CLI agent example
import json
import subprocess

class ObsidianSearch:
    def __init__(self):
        self.proc = subprocess.Popen(
            ['obsidian', '--search-rpc'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
    
    def search(self, query: str, max_results: int = 10):
        request = {
            'jsonrpc': '2.0',
            'method': 'search',
            'params': {'query': query, 'options': {'maxResults': max_results}},
            'id': 1
        }
        
        self.proc.stdin.write(json.dumps(request) + '\n')
        self.proc.stdin.flush()
        
        response = json.loads(self.proc.stdout.readline())
        return response['result']

# Usage
search = ObsidianSearch()
results = search.search("quantum computing")
for result in results:
    print(f"{result['id']}: {result['score']:.3f}")
```

## Performance Characteristics

### Embedding Generation

- **Speed**: 50-100ms per query (WebAssembly)
- **Memory**: ~50MB for model
- **Offline**: Fully local, no network needed

### Vector Search

- **Speed**: <1ms for 10k vectors (HNSW)
- **Memory**: ~15MB per 10k documents (384-dim)
- **Scalability**: Linear with document count

### Lexical Search

- **Speed**: <5ms for 100k documents (BM25)
- **Memory**: ~5MB per 10k documents
- **Scalability**: Linear with document count

### Total System

- **Index Time**: ~100ms per document
- **Search Time**: ~100ms end-to-end
- **Memory Usage**: ~100MB for 10k documents
- **Disk Usage**: Optional (~50MB for 10k documents)

## Comparison with Current Implementation

| Aspect | Current System | Recommended System |
|--------|---------------|-------------------|
| **Dependencies** | 10+ packages, LangChain ecosystem | 2 packages total |
| **External Services** | Optional Ollama/LM Studio | None |
| **Bundle Size** | ~5MB+ | ~700KB |
| **Setup Complexity** | API keys or Ollama setup | Zero config |
| **CLI Access** | Custom protocol | Standard JSON-RPC |
| **Embedding Source** | External APIs or Ollama | Built-in WebAssembly |
| **Performance** | Variable (depends on external) | Consistent (~100ms) |
| **Offline** | Requires Ollama for local | Fully offline |

## Migration Path

For existing users who want to migrate to the simplified system:

### Step 1: Add New System (Parallel)

```typescript
// Add new system alongside existing
class ModernSemanticSearch {
  // New implementation
}

// Keep old system for compatibility
class LegacySemanticSearch {
  // Existing implementation
}
```

### Step 2: Feature Flag

```typescript
// User can choose system in settings
const searchEngine = settings.useModernSearch 
  ? new ModernSemanticSearch()
  : new LegacySemanticSearch();
```

### Step 3: Gradual Rollout

1. Release with opt-in flag
2. Monitor performance and feedback
3. Make modern system default
4. Eventually deprecate legacy system

## Security Considerations

### Local-Only Benefits

1. **No API Keys**: No risk of key leakage
2. **No Network**: No data sent externally
3. **No External Process**: Fewer attack vectors
4. **Sandboxed**: Runs in plugin context

### JSON-RPC Security

1. **Stdio Only**: No network exposure
2. **Input Validation**: Validate all RPC params
3. **Rate Limiting**: Prevent abuse
4. **Permissions**: Respect Obsidian's plugin permissions

## Conclusion

This simplified implementation provides:

✅ **Local-only operation** - Everything runs in the plugin  
✅ **No external dependencies** - Only 2 small packages needed  
✅ **CLI accessible** - Standard JSON-RPC protocol  
✅ **Zero configuration** - Works out-of-the-box  
✅ **Good performance** - ~100ms search latency  
✅ **Small footprint** - ~700KB bundle size  
✅ **Easy maintenance** - Simple codebase  

This approach trades some flexibility (fewer embedding models, simpler algorithms) for significant gains in simplicity, reliability, and user experience.

## Additional Resources

**Transformers.js Documentation**: https://huggingface.co/docs/transformers.js  
**HNSW Algorithm**: https://arxiv.org/abs/1603.09320  
**BM25 Reference**: https://en.wikipedia.org/wiki/Okapi_BM25  
**JSON-RPC Specification**: https://www.jsonrpc.org/specification  
**Reciprocal Rank Fusion**: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

---

**Document Purpose**: This is a recommendation for a simplified approach. The actual implementation in this repository may differ. See `SEMANTIC_SEARCH_IMPLEMENTATION.md` for details on the current implementation.
