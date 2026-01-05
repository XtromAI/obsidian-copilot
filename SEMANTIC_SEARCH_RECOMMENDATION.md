# Semantic Search: Simplified Modern Implementation

## Overview

This document provides recommendations for implementing a simplified, modern semantic search system for a **CLI-based AI Agent Client** that integrates with Obsidian to provide AI assistant capabilities.

**Target Audience**: Developers building CLI tools that need to search Obsidian vaults for AI-powered assistance.

**Use Case**: An external CLI agent (separate from this Obsidian Copilot plugin) that:
- Connects to Obsidian to provide AI assistance
- Needs to search the vault to provide context for AI responses
- Wants a simple, local-only implementation
- Requires minimal dependencies and easy deployment

### Design Priorities

- **Local-only operation** (no external API dependencies)
- **Standalone CLI tool** (runs independently of Obsidian plugin)
- **Minimal dependencies** (keep the tool lightweight)
- **Simple deployment** (easy for users to install and use)

## Design Philosophy

### Core Principles

1. **Everything Runs Locally**: No cloud APIs, no external services
2. **Standalone Tool**: Independent CLI application that reads Obsidian vaults
3. **Zero Configuration**: Works out-of-the-box with sensible defaults
4. **Obsidian Integration**: Reads vault files directly, no plugin required

## Recommended Architecture

### High-Level Design

```
┌─────────────────────────────────────────┐
│    Your CLI Agent Tool (Separate App)   │
│  ┌───────────────────────────────────┐  │
│  │  AI Assistant Logic               │  │
│  └───────────┬───────────────────────┘  │
│              │                           │
│  ┌───────────▼───────────────────────┐  │
│  │  Semantic Search Module           │  │
│  │  - Lexical (BM25)                 │  │
│  │  - Semantic (Local Embeddings)    │  │
│  └───────────┬───────────────────────┘  │
│              │                           │
│  ┌───────────▼───────────────────────┐  │
│  │  Vault File Reader                │  │
│  │  - Reads .md files                │  │
│  │  - Parses frontmatter             │  │
│  └───────────────────────────────────┘  │
└─────────────┬───────────────────────────┘
              │ Direct file access
              ↓
┌─────────────────────────────────────────┐
│       Obsidian Vault (File System)      │
│  - Notes/*.md                            │
│  - Daily/*.md                            │
│  - Projects/*.md                         │
└─────────────────────────────────────────┘
```

**Key Difference**: Your CLI tool directly reads the Obsidian vault files from the filesystem, no need for plugin communication.

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

### 5. Vault File Reader

**Recommendation**: Direct filesystem access to read Obsidian vault.

```typescript
import * as fs from 'fs';
import * as path from 'path';
import matter from 'gray-matter';  // For frontmatter parsing

class VaultReader {
  constructor(private vaultPath: string) {}
  
  // Find all markdown files in vault
  async getAllMarkdownFiles(): Promise<string[]> {
    const files: string[] = [];
    
    const walk = async (dir: string) => {
      const entries = await fs.promises.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory()) {
          // Skip .obsidian and .trash folders
          if (!entry.name.startsWith('.')) {
            await walk(fullPath);
          }
        } else if (entry.name.endsWith('.md')) {
          files.push(fullPath);
        }
      }
    };
    
    await walk(this.vaultPath);
    return files;
  }
  
  // Read and parse a markdown file
  async readNote(filePath: string): Promise<Note> {
    const content = await fs.promises.readFile(filePath, 'utf-8');
    const { data: frontmatter, content: body } = matter(content);
    
    return {
      path: path.relative(this.vaultPath, filePath),
      content: body,
      frontmatter,
      title: path.basename(filePath, '.md')
    };
  }
  
  // Watch for file changes (optional)
  watchVault(callback: (event: string, filename: string) => void) {
    fs.watch(this.vaultPath, { recursive: true }, (event, filename) => {
      if (filename && filename.endsWith('.md')) {
        callback(event, path.join(this.vaultPath, filename));
      }
    });
  }
}

interface Note {
  path: string;
  content: string;
  frontmatter: any;
  title: string;
}
```

**Benefits**:
- No plugin dependency
- Direct access to all vault files
- Can read metadata from frontmatter
- Simple to implement
- Works even when Obsidian is closed

### 6. Unified Search Engine

**Recommendation**: Combine lexical and semantic search with simple fusion.

```typescript
class UnifiedSearch {
  private lexical: BM25Search;
  private semantic: VectorIndex;
  private embedder: LocalEmbedding;
  private chunker: SimpleChunker;
  private vaultReader: VaultReader;
  
  constructor(vaultPath: string) {
    this.lexical = new BM25Search();
    this.semantic = new VectorIndex(384);
    this.embedder = new LocalEmbedding();
    this.chunker = new SimpleChunker();
    this.vaultReader = new VaultReader(vaultPath);
  }
  
  // Index entire vault
  async indexVault() {
    const files = await this.vaultReader.getAllMarkdownFiles();
    
    for (const filePath of files) {
      const note = await this.vaultReader.readNote(filePath);
      await this.indexDocument(note.path, note.content);
    }
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
  "hnswlib-wasm": "^1.0.0",             // Vector search (~200KB)
  "gray-matter": "^4.0.3"               // Frontmatter parsing (~50KB)
}
```

**Total Bundle Impact**: ~750KB (acceptable for CLI tool)

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

### For CLI Tool Users

**Simple Usage**: Point the tool at your Obsidian vault.

```bash
# Install your CLI tool
npm install -g your-ai-agent

# Initialize with vault path
your-ai-agent init ~/Documents/ObsidianVault

# Use AI assistant (search happens automatically)
your-ai-agent ask "What notes do I have about machine learning?"
```

### For CLI Tool Developers

**Implementation Example**:

```typescript
// Main CLI application
import { UnifiedSearch } from './search';
import { Command } from 'commander';

const program = new Command();
let searchEngine: UnifiedSearch;

program
  .command('init <vaultPath>')
  .description('Initialize with Obsidian vault')
  .action(async (vaultPath: string) => {
    console.log('Indexing vault...');
    searchEngine = new UnifiedSearch(vaultPath);
    await searchEngine.indexVault();
    console.log('Done! Vault indexed.');
  });

program
  .command('ask <question>')
  .description('Ask AI assistant a question')
  .action(async (question: string) => {
    // Search vault for relevant context
    const results = await searchEngine.search(question, { maxResults: 5 });
    
    // Build context for AI
    const context = results
      .map(r => `Source: ${r.id}\n${r.content}`)
      .join('\n\n---\n\n');
    
    // Send to AI with context (your AI logic here)
    const answer = await yourAIFunction(question, context);
    console.log(answer);
  });

program.parse();
```

**Python Example**:

```python
# Python CLI tool
import sys
from pathlib import Path
from your_search import UnifiedSearch

class ObsidianAIAgent:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.search = UnifiedSearch(vault_path)
    
    async def init(self):
        """Index the vault"""
        print("Indexing vault...")
        await self.search.index_vault()
        print("Done!")
    
    async def ask(self, question: str):
        """Answer a question using vault context"""
        # Search vault
        results = await self.search.search(question, max_results=5)
        
        # Build context
        context = "\n\n---\n\n".join([
            f"Source: {r['id']}\n{r['content']}"
            for r in results
        ])
        
        # Your AI logic here
        answer = await your_ai_function(question, context)
        print(answer)

# Usage
if __name__ == "__main__":
    agent = ObsidianAIAgent(sys.argv[1])
    asyncio.run(agent.init())
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

## Comparison with Obsidian Copilot Plugin

| Aspect | Obsidian Copilot Plugin | Recommended CLI Tool |
|--------|------------------------|---------------------|
| **Type** | Obsidian plugin | Standalone CLI application |
| **Dependencies** | 10+ packages, LangChain ecosystem | 3 packages total |
| **External Services** | Optional Ollama/LM Studio | None |
| **Bundle Size** | ~5MB+ | ~750KB |
| **Setup Complexity** | API keys or Ollama setup | Zero config |
| **Vault Access** | Via Obsidian API | Direct filesystem access |
| **Embedding Source** | External APIs or Ollama | Built-in WebAssembly |
| **Performance** | Variable (depends on external) | Consistent (~100ms) |
| **Offline** | Requires Ollama for local | Fully offline |
| **Obsidian Required** | Yes | No (reads files directly) |

## Development Roadmap

Building this as a standalone CLI tool:

### Week 1: Core Search

```bash
your-agent/
├── src/
│   ├── search/
│   │   ├── bm25.ts          # Lexical search
│   │   ├── vector.ts        # Vector index
│   │   ├── embeddings.ts    # Transformers.js wrapper
│   │   ├── chunker.ts       # Simple chunking
│   │   └── unified.ts       # Combines both
│   ├── vault/
│   │   └── reader.ts        # Vault file reader
│   └── index.ts             # CLI entry point
├── package.json
└── README.md
```

### Week 2: CLI Interface

```bash
# Commands to implement
your-agent init <vault-path>     # Index vault
your-agent search <query>        # Test search
your-agent ask <question>        # AI assistant
your-agent reindex              # Rebuild index
```

### Week 3: AI Integration

- Integrate with your preferred AI model
- Add conversation context management
- Implement result ranking
- Add caching for repeated queries

## Security Considerations

### Local-Only Benefits

1. **No API Keys**: No risk of key leakage
2. **No Network**: No data sent externally
3. **Direct File Access**: Read-only by default
4. **Offline**: No external dependencies

### Vault Access Security

1. **Read-Only**: Only read vault files, don't modify
2. **Respect .obsidian**: Skip configuration folders
3. **Path Validation**: Validate vault path is legitimate
4. **Permissions**: Follow OS file permissions

## Conclusion

This simplified implementation provides a **standalone CLI tool** for AI agents with:

✅ **Local-only operation** - Everything runs in your CLI tool  
✅ **No external dependencies** - Only 3 small packages needed  
✅ **Direct vault access** - Read Obsidian files directly from filesystem  
✅ **Zero configuration** - Works out-of-the-box  
✅ **Good performance** - ~100ms search latency  
✅ **Small footprint** - ~750KB bundle size  
✅ **Easy maintenance** - Simple codebase  
✅ **No Obsidian required** - Can run even when Obsidian is closed

This approach is ideal for building CLI-based AI assistants that need to search Obsidian vaults without requiring the Obsidian Copilot plugin.

## Additional Resources

**Transformers.js Documentation**: https://huggingface.co/docs/transformers.js  
**HNSW Algorithm**: https://arxiv.org/abs/1603.09320  
**BM25 Reference**: https://en.wikipedia.org/wiki/Okapi_BM25  
**Reciprocal Rank Fusion**: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf  
**gray-matter (Frontmatter)**: https://github.com/jonschlinkert/gray-matter

---

**Document Purpose**: This is a recommendation for building a separate CLI-based AI agent tool that needs semantic search capabilities for Obsidian vaults. This is NOT a recommendation for modifying the Obsidian Copilot plugin in this repository. See `SEMANTIC_SEARCH_IMPLEMENTATION.md` for details on how the Obsidian Copilot plugin currently implements search.
