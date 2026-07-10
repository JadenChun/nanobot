# Unified Context Management Refactor - Implementation Plan

## Overview

Refactor nanobot's context management from fragmented mechanisms into a unified `ContextBudgetManager` that manages all dynamic content (memory + history + tool results) as a single "active context" pool with a shared budget.

## Goals

1. **Unified budget**: Single budget for all dynamic context (memory + history + tool results)
2. **Automatic compression**: Compress oldest content when over budget
3. **Memory as active context**: Move MEMORY.md from system prompt into managed message pool
4. **Preserve sharing**: Maintain cron job ↔ interactive chat memory sharing
5. **Backward compatible**: Existing code paths continue to work

## Non-Goals

- Change the LLM provider interface
- Modify tool execution logic
- Change session persistence format (JSONL)
- Remove HISTORY.md (still append-only log)

---

## Architecture

### New Components

```
┌─────────────────────────────────────────────────────────────┐
│ ContextBudgetManager (nanobot/agent/context_budget.py)     │
│                                                             │
│  Responsibilities:                                          │
│   • Measure total context tokens                            │
│   • Apply reduction strategies when over budget             │
│   • Coordinate with MemoryStore for persistence             │
│                                                             │
│  Strategies (in order):                                     │
│   1. Compact old tool results (lossy, fast)                 │
│   2. Drop old turn pairs (destructive, fast)                │
│   3. Compress memory sections (LLM call, slow)            │
│   4. Truncate oldest memory (last resort)                   │
└─────────────────────────────────────────────────────────────┘
```

### Modified Components

| Component | Change |
|-----------|--------|
| `ContextBuilder` | Remove MEMORY.md from `build_system_prompt()`. Add `build_active_context_message()` to create `[Past Knowledge]` message. |
| `AgentLoop` | Instantiate `ContextBudgetManager`. Call it pre-flight and mid-loop instead of scattered trimming. |
| `AgentRunner` | Remove `_trim_context_to_budget()`. Delegate to `ContextBudgetManager`. |
| `MemoryStore` | Add `compress_if_needed()` method. Keep existing `consolidate()`. |
| `clear_old_tool_results()` | Keep as standalone function, called by `ContextBudgetManager`. |

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Session Start                                               │
├─────────────────────────────────────────────────────────────┤
│ 1. Load MEMORY.md from disk                                 │
│ 2. ContextBuilder.build_messages():                         │
│    - system_prompt (no memory)                              │
│    - [Past Knowledge] message (MEMORY.md content)           │
│    - history                                                │
│    - user_message                                           │
│ 3. ContextBudgetManager.enforce_budget(messages)            │
│    - Measure tokens                                         │
│    - Apply strategies if over budget                        │
│ 4. Return messages for LLM call                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent Loop (each iteration)                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. LLM call                                                 │
│ 2. Tool execution                                           │
│ 3. Append assistant + tool messages                         │
│ 4. ContextBudgetManager.enforce_budget(messages)            │
│    - Compact old tool results                               │
│    - Drop old turns if needed                               │
│    - Compress memory if needed                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Session End                                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. ContextBudgetManager.consolidate(messages)               │
│    - Compress active context                                │
│    - Update MEMORY.md                                       │
│    - Append to HISTORY.md                                   │
│ 2. Save session                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Design

### 1. ContextBudget Dataclass

**File**: `nanobot/agent/context_budget.py`

```python
@dataclass
class ContextBudget:
    """Configuration for context budget management."""
    
    # Total token limit (from config.maxTokens.input)
    max_tokens: int
    
    # Reserved for output (from config.maxTokens.output)
    output_reserve: int = 2000
    
    # Safety buffer for estimation drift (tokens)
    # This is the "headroom" that protects against estimation errors.
    # With accurate provider token counting, this can be smaller.
    safety_buffer: int = 1024
    
    # Compression thresholds (as fraction of available budget)
    # trigger_ratio = 1.0 means: only compress when we hit the limit.
    # This maximizes context utilization but requires accurate token counting.
    trigger_ratio: float = 1.0  # Compress at 100% of budget
    
    # Target after compression (as fraction of available budget)
    # 0.2 means: compress down to 20% to leave room for new content
    target_ratio: float = 0.2
    
    # Memory compression thresholds (absolute tokens)
    memory_max_tokens: int = 1500   # Max tokens for MEMORY.md content
    memory_target_tokens: int = 1000  # Compress to this if over max
    
    @property
    def available_budget(self) -> int:
        """Tokens available for active context (memory + history + tools)."""
        return self.max_tokens - self.output_reserve - self.safety_buffer
    
    @property
    def trigger_tokens(self) -> int:
        """Token count that triggers compression.
        
        With trigger_ratio=1.0, this equals available_budget.
        The safety_buffer protects against estimation drift.
        """
        return int(self.available_budget * self.trigger_ratio)
    
    @property
    def target_tokens(self) -> int:
        """Target token count after compression."""
        return int(self.available_budget * self.target_ratio)
```

**Key Design Decision**: Using `trigger_ratio = 1.0` maximizes context utilization.

- **Pros**: Uses full budget, no wasted space
- **Cons**: Requires accurate token counting to avoid exceeding actual limit
- **Mitigation**: The `safety_buffer` (1024 tokens) absorbs estimation errors

**Token Counting Accuracy**:

The accuracy of token counting determines how safely we can run at 100% budget:

| Method | Accuracy | Recommendation |
|--------|----------|----------------|
| Heuristic (4 chars/token) | ±20% | Not safe for 100% budget |
| Tiktoken (OpenAI tokenizer) | ±5% | Acceptable for 100% budget |
| Provider API (actual count) | Exact | Best, use when available |

**Implementation**: Use `estimate_prompt_tokens_chain()` which:
1. Tries provider's tokenizer (most accurate)
2. Falls back to tiktoken if provider doesn't support it
3. Falls back to heuristic as last resort

**Warning**: If token counting is inaccurate, increase `safety_buffer` or lower `trigger_ratio` to 0.95.

### 2. ContextBudgetManager Class

**File**: `nanobot/agent/context_budget.py`

```python
class ContextBudgetManager:
    """Unified context budget management."""
    
    def __init__(
        self,
        budget: ContextBudget,
        memory_store: MemoryStore,
        provider: LLMProvider,
        model: str,
        tool_registry: ToolRegistry,
    ):
        self.budget = budget
        self.memory = memory_store
        self.provider = provider
        self.model = model
        self.tools = tool_registry
    
    def enforce_budget(
        self,
        messages: list[dict[str, Any]],
        *,
        preserve_last_n_turns: int = 2,
    ) -> list[dict[str, Any]]:
        """Ensure messages fit within budget.
        
        Applies reduction strategies in order:
        1. Compact old tool results
        2. Drop old turn pairs
        3. Compress memory sections
        
        Returns modified messages (may be same list, mutated).
        """
        tokens = self._measure_tokens(messages)
        if tokens <= self.budget.trigger_tokens:
            return messages
        
        logger.info(
            "Context over budget: {} > {} tokens. Applying reduction.",
            tokens, self.budget.trigger_tokens
        )
        
        # Strategy 1: Compact old tool results
        self._compact_tool_results(messages, preserve_last_n_turns)
        tokens = self._measure_tokens(messages)
        if tokens <= self.budget.target_tokens:
            return messages
        
        # Strategy 2: Drop old turn pairs
        self._drop_old_turns(messages, preserve_last_n_turns)
        tokens = self._measure_tokens(messages)
        if tokens <= self.budget.target_tokens:
            return messages
        
        # Strategy 3: Compress memory (LLM call)
        self._compress_memory(messages)
        
        return messages
    
    def consolidate(self, messages: list[dict[str, Any]]) -> None:
        """Consolidate active context to MEMORY.md at session end."""
        # Extract conversation content (skip system, past knowledge)
        conversation = [
            m for m in messages
            if m.get("role") not in ("system",)
            and not m.get("content", "").startswith("[Past Knowledge]")
        ]
        
        if conversation:
            self.memory.consolidate(conversation, self.provider, self.model)
    
    # --- Private methods ---
    
    def _measure_tokens(self, messages: list[dict]) -> int:
        """Measure total tokens in messages + tool defs."""
        tool_defs = self.tools.get_definitions()
        tokens, _ = estimate_prompt_tokens_chain(
            self.provider, self.model, messages, tool_defs
        )
        return tokens
    
    def _compact_tool_results(
        self,
        messages: list[dict],
        preserve_last_n: int,
    ) -> None:
        """Compact old tool result content."""
        # Find tool result messages
        tool_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "tool"
        ]
        
        if len(tool_indices) <= preserve_last_n:
            return
        
        to_compact = tool_indices[:-preserve_last_n]
        for idx in to_compact:
            content = messages[idx].get("content", "")
            if not content.startswith("[compacted"):
                compacted = _compact_tool_result_content(content)
                messages[idx] = {**messages[idx], "content": compacted}
    
    def _drop_old_turns(
        self,
        messages: list[dict],
        preserve_last_n: int,
    ) -> None:
        """Drop old assistant+tool turn pairs."""
        # Identify turns (assistant + following tool results)
        turns = []
        i = 1  # Skip system message
        while i < len(messages):
            if messages[i].get("role") == "assistant":
                start = i
                i += 1
                while i < len(messages) and messages[i].get("role") == "tool":
                    i += 1
                turns.append((start, i - 1))
            else:
                i += 1
        
        if len(turns) <= preserve_last_n:
            return
        
        # Drop oldest turns (keep last N)
        to_drop = turns[:-preserve_last_n]
        for start, end in reversed(to_drop):
            del messages[start:end + 1]
    
    def _compress_memory(self, messages: list[dict]) -> None:
        """Compress [Past Knowledge] message if too large."""
        # Find [Past Knowledge] message
        past_knowledge_idx = None
        for i, m in enumerate(messages):
            content = m.get("content", "")
            if content.startswith("[Past Knowledge]"):
                past_knowledge_idx = i
                break
        
        if past_knowledge_idx is None:
            return
        
        content = messages[past_knowledge_idx].get("content", "")
        # Rough token estimate (4 chars per token)
        estimated_tokens = len(content) // 4
        
        if estimated_tokens <= self.budget.memory_max_tokens:
            return
        
        logger.info(
            "Memory too large: ~{} tokens. Compressing.",
            estimated_tokens
        )
        
        # Call LLM to compress
        compressed = self._llm_compress_memory(content)
        messages[past_knowledge_idx]["content"] = compressed
    
    def _llm_compress_memory(self, content: str) -> str:
        """Use LLM to compress memory content."""
        prompt = f"""Compress this agent memory to ~{self.budget.memory_target_tokens} tokens.
Preserve: key facts, recent events, important decisions.
Remove: redundant details, old events, verbose descriptions.

Current memory:
{content}

Return compressed memory, starting with [Past Knowledge]."""
        
        response = self.provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=self.budget.memory_target_tokens + 200,
        )
        
        compressed = response.content or ""
        if not compressed.startswith("[Past Knowledge]"):
            compressed = f"[Past Knowledge]\n{compressed}"
        
        return compressed
```

### 3. ContextBuilder Changes

**File**: `nanobot/agent/context.py`

```python
def build_system_prompt(
    self,
    skill_names: list[str] | None = None,
    tool_names: set[str] | None = None,
) -> str:
    """Build system prompt WITHOUT memory."""
    parts = [self._get_identity(tool_names)]
    
    bootstrap = self._load_bootstrap_files()
    if bootstrap:
        parts.append(bootstrap)
    
    # ... skills, planning section ...
    
    # NOTE: Memory is NOT included here anymore
    
    return "\n\n---\n\n".join(parts)


def build_active_context_message(self, memory_content: str) -> dict[str, Any]:
    """Build [Past Knowledge] message from MEMORY.md content."""
    if not memory_content:
        return {
            "role": "user",
            "content": "[Past Knowledge]\nNo prior knowledge.",
        }
    
    return {
        "role": "user",
        "content": f"[Past Knowledge]\n{memory_content}",
    }


def build_messages(
    self,
    history: list[dict[str, Any]],
    current_message: str,
    # ... other params ...
    include_memory: bool = True,
) -> list[dict[str, Any]]:
    """Build message list with [Past Knowledge] as managed message."""
    system_prompt = self.build_system_prompt(skill_names, tool_names)
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add [Past Knowledge] if requested
    if include_memory:
        memory_content = self.memory.read_long_term()
        if memory_content:
            past_msg = self.build_active_context_message(memory_content)
            messages.append(past_msg)
    
    # Add history
    messages.extend(history)
    
    # Add current message
    messages.append({
        "role": current_role,
        "content": self._build_user_content(current_message, media),
    })
    
    return messages
```

### 4. AgentLoop Changes

**File**: `nanobot/agent/loop.py`

```python
def __init__(self, ...):
    # ... existing init ...
    
    # Create budget manager
    self.budget_manager = ContextBudgetManager(
        budget=ContextBudget(
            max_tokens=self.max_tokens.input,
            output_reserve=self.max_tokens.output,
        ),
        memory_store=self.context.memory,
        provider=self.provider,
        model=self.model,
        tool_registry=self.tools,
    )


async def _process_message(self, msg, ...):
    # Build messages (now includes [Past Knowledge])
    messages = self.context.build_messages(
        history=history,
        current_message=current_message,
        include_memory=True,  # Memory is now a managed message
    )
    
    # Pre-flight budget enforcement
    messages = self.budget_manager.enforce_budget(messages)
    
    # Run agent loop
    result = await self._run_agent_loop(messages, ...)
    
    # Post-run consolidation
    self.budget_manager.consolidate(result.messages)
    
    return result
```

### 5. AgentRunner Changes

**File**: `nanobot/agent/runner.py`

Remove `_trim_context_to_budget()` method. The budget manager is called by AgentLoop before passing messages to runner.

Alternatively, pass budget_manager to runner and call it mid-loop:

```python
async def run(self, spec: AgentRunSpec) -> AgentRunResult:
    for iteration in range(spec.max_iterations):
        # Mid-loop budget enforcement
        if spec.budget_manager and iteration > 0:
            messages = spec.budget_manager.enforce_budget(messages)
        
        # ... rest of loop ...
```

### 6. MemoryStore Changes

**File**: `nanobot/agent/memory.py`

Keep existing methods. Add:

```python
def compress_to_target(
    self,
    target_tokens: int,
    provider: LLMProvider,
    model: str,
) -> str:
    """Compress MEMORY.md to target token count.
    
    Returns compressed content (does not write to disk).
    """
    current = self.read_long_term()
    current_tokens = len(current) // 4  # rough estimate
    
    if current_tokens <= target_tokens:
        return current
    
    # Use LLM to compress
    prompt = f"""Compress this memory to ~{target_tokens} tokens.
Preserve: key facts, recent events, important decisions.
Remove: redundant details, old events, verbose descriptions.

Current memory:
{current}

Return compressed memory."""
    
    response = provider.chat_with_retry(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=target_tokens + 200,
    )
    
    return response.content or current
```

---

## Migration Path

### Phase 1: Add New Components (Non-Breaking)

1. Create `context_budget.py` with `ContextBudget` and `ContextBudgetManager`
2. Add `build_active_context_message()` to `ContextBuilder`
3. Add `compress_to_target()` to `MemoryStore`

**Tests**: Unit tests for new classes (no integration yet)

### Phase 2: Wire Up (Opt-In)

1. Add `use_unified_budget: bool = False` flag to `AgentLoop`
2. When flag is True, use `ContextBudgetManager` instead of scattered logic
3. Keep old code paths for backward compatibility

**Tests**: Integration tests with flag enabled

### Phase 3: Default On

1. Set `use_unified_budget: bool = True` as default
2. Remove old scattered trimming logic
3. Update documentation

**Tests**: Full regression suite

### Phase 4: Cleanup

1. Remove `use_unified_budget` flag (always on)
2. Remove dead code (`_trim_context_to_budget`, etc.)

---

## Testing Strategy

### Unit Tests

- `test_context_budget.py`:
  - `test_budget_calculation`: Verify available_budget, trigger_tokens, target_tokens
  - `test_compact_tool_results`: Verify old results are compacted
  - `test_drop_old_turns`: Verify turns are dropped correctly
  - `test_compress_memory`: Mock LLM, verify compression is called

### Integration Tests

- `test_unified_context.py`:
  - `test_cron_job_flow`: Verify cron job runs, compresses, saves to MEMORY.md
  - `test_chat_flow`: Verify chat loads memory, compresses when over budget
  - `test_memory_sharing`: Verify cron job updates are visible to chat

### Regression Tests

- Existing `test_runner_context_trim.py` (may need updates)
- Existing `test_context_prompt_cache.py`
- Existing memory consolidation tests

---

## Risk Mitigation

### Risk 1: LLM Compression Latency

**Problem**: Memory compression requires LLM call (slow on resource-constrained devices)

**Mitigation**:
- Only compress when absolutely necessary (over budget)
- Use smaller/faster model for compression (if available)
- Cache compressed results to avoid re-compression

### Risk 2: Information Loss

**Problem**: Compression loses details

**Mitigation**:
- HISTORY.md is append-only (details still searchable)
- Compression prompt emphasizes preserving key facts
- Log what was compressed for debugging

### Risk 3: Backward Compatibility

**Problem**: Existing code expects MEMORY.md in system prompt

**Mitigation**:
- Phase 2: Opt-in flag
- Phase 3: Monitor for issues before removing old code

### Risk 4: Provider Differences

**Problem**: Some providers may not support multiple user messages or custom markers

**Mitigation**:
- Test with oMLX (primary provider)
- Fallback: If `[Past Knowledge]` causes issues, merge into system prompt

---

## Design Decisions (Resolved)

### 1. Memory Injection Format

**Decision**: `[Past Knowledge]` will be a **user message**.

**Rationale**:
- Fits naturally into conversation flow
- Compatible with most providers
- Can be compressed/managed like other user messages

**Implementation**:
```python
def build_active_context_message(self, memory_content: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": f"[Past Knowledge]\n{memory_content}",
    }
```

### 2. Compression Model

**Decision**: Use the **same model** as the main agent.

**Rationale**:
- Simplicity (no need to configure/manage multiple models)
- Consistency (compression understands the same context as main agent)
- For resource-constrained devices, compression only triggers when necessary

**Trade-off**: Slower than using a smaller model, but acceptable since compression is rare.

### 3. Consolidation Timing

**Decision**: Consolidate **only when memory hits the limit**, not every session end.

**Rationale**:
- Maximizes context utilization (use 100% of budget)
- Avoids unnecessary LLM calls for compression
- The 1024 token safety buffer protects against estimation errors

**Implementation**:
- `trigger_ratio = 1.0` (compress at 100% of budget)
- `safety_buffer = 1024` (absorbs estimation drift)
- Compression triggers only when `tokens >= trigger_tokens`

**Requirement**: Accurate token counting. See "Token Counting Accuracy" section above.

### 4. HISTORY.md Format

**Decision**: Add **source tags** to HISTORY.md entries.

**Format**:
```
[2025-01-15 14:30] [source=cron] Cron job completed: checked email, found 3 new messages
[2025-01-15 15:45] [source=chat] User asked about email, summarized cron job findings
```

**Implementation**:
```python
def append_history(self, entry: str, source: str = "unknown") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    formatted = f"[{ts}] [source={source}] {entry}"
    with open(self.history_file, "a", encoding="utf-8") as f:
        f.write(formatted + "\n\n")
```

**Benefits**:
- Easy to filter by source (grep for cron vs chat)
- Helps debug memory sharing between cron and chat
- Minimal change to existing format

---

## Open Questions (For Future Consideration)

1. **Configurable budget ratios**: Should `trigger_ratio` and `target_ratio` be configurable in `config.json`?
   - Current plan: Hardcoded defaults, can be made configurable later

2. **Multi-model compression**: If a smaller/faster model is available, should we use it for compression?
   - Current plan: Use same model for simplicity
   - Future: Could add `compression_model` config option

3. **Compression caching**: Should compressed memory be cached to avoid re-compression?
   - Current plan: No caching (compression is rare)
   - Future: Could cache compressed results keyed by content hash

---

## Success Criteria

1. Cron jobs complete without timeout on 16GB device
2. Interactive chat can see cron job results via MEMORY.md
3. Total context stays within `maxTokens.input`
4. No regression in existing functionality
5. Code is simpler (fewer scattered trimming mechanisms)

---

## Timeline Estimate

- Phase 1 (New components): 2-3 hours
- Phase 2 (Wire up, opt-in): 2-3 hours
- Phase 3 (Default on): 1-2 hours
- Phase 4 (Cleanup): 1 hour
- Testing: 2-3 hours

**Total**: 8-12 hours

---

## Next Steps

1. **Review this updated plan** (with resolved design decisions)
2. **Confirm ready to start** Phase 1 implementation
3. **Implement Phase 1**: Create new components (`ContextBudget`, `ContextBudgetManager`)
4. **Test Phase 1**: Unit tests for new classes
5. **Proceed to Phase 2**: Wire up with opt-in flag
