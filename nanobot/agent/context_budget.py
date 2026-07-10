"""Unified context budget management for nanobot agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.utils.helpers import estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.agent.memory import MemoryStore
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.providers.base import LLMProvider


@dataclass
class ContextBudget:
    """Configuration for context budget management.
    
    Defines token limits and compression thresholds for the unified
    active context model.
    
    Attributes:
        max_tokens: Total token limit (from config.maxTokens.input)
        output_reserve: Tokens reserved for output (from config.maxTokens.output)
        safety_buffer: Extra headroom for estimation drift (tokens)
        trigger_ratio: Fraction of budget that triggers compression (1.0 = 100%)
        target_ratio: Target fraction after compression (0.2 = 20%)
        memory_max_tokens: Max tokens for MEMORY.md content
        memory_target_tokens: Compress memory to this if over max
    """
    
    max_tokens: int
    output_reserve: int = 2000
    safety_buffer: int = 1024
    
    # trigger_ratio = 1.0 means: only compress when we hit the limit.
    # This maximizes context utilization but requires accurate token counting.
    # The safety_buffer protects against estimation errors.
    trigger_ratio: float = 1.0
    
    # Target after compression (as fraction of available budget)
    # 0.2 means: compress down to 20% to leave room for new content
    target_ratio: float = 0.2
    
    # Memory compression thresholds (absolute tokens)
    memory_max_tokens: int = 1500
    memory_target_tokens: int = 1000
    
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


class ContextBudgetManager:
    """Unified context budget management.
    
    Manages all dynamic content (memory + history + tool results) as a single
    "active context" pool with a shared budget. Applies reduction strategies
    in hierarchical order when the context exceeds the budget.
    
    Reduction strategies (in order):
    1. Compact old tool results (lossy, fast)
    2. Drop old turn pairs (destructive, fast)
    3. Compress memory sections (LLM call, slow)
    4. Truncate oldest memory (last resort)
    """
    
    _COMPACTED_PLACEHOLDER = "[compacted to save context]"
    _COMPACTED_HEAD_LINES = 6
    _COMPACTED_TAIL_LINES = 4
    _COMPACTED_MAX_CHARS = 700
    
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
    
    async def enforce_budget(
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
        
        Args:
            messages: Message list to enforce budget on
            preserve_last_n_turns: Number of recent turns to preserve
            
        Returns:
            Modified messages that fit within budget
        """
        tokens = self._measure_tokens(messages)
        if tokens <= self.budget.trigger_tokens:
            return messages
        
        logger.info(
            "Context over budget: {} > {} tokens (budget {}). Applying reduction.",
            tokens,
            self.budget.trigger_tokens,
            self.budget.available_budget,
        )
        
        # Strategy 1: Compact old tool results
        self._compact_tool_results(messages, preserve_last_n_turns)
        tokens = self._measure_tokens(messages)
        if tokens <= self.budget.target_tokens:
            logger.info("Compact tool results reduced to {} tokens", tokens)
            return messages
        
        # Strategy 2: Drop old turn pairs
        self._drop_old_turns(messages, preserve_last_n_turns)
        tokens = self._measure_tokens(messages)
        if tokens <= self.budget.target_tokens:
            logger.info("Drop old turns reduced to {} tokens", tokens)
            return messages
        
        # Strategy 3: Compress memory (LLM call)
        await self._compress_memory(messages)
        tokens = self._measure_tokens(messages)
        logger.info("Compress memory reduced to {} tokens", tokens)
        
        return messages
    
    def consolidate(
        self,
        messages: list[dict[str, Any]],
        source: str = "unknown",
    ) -> None:
        """Consolidate active context to MEMORY.md at session end.
        
        Args:
            messages: Message list to consolidate
            source: Source tag for HISTORY.md entries (e.g., "cron", "chat")
        """
        # Extract conversation content (skip system, past knowledge)
        conversation = [
            m for m in messages
            if m.get("role") not in ("system",)
            and not m.get("content", "").startswith("[Past Knowledge]")
        ]
        
        if conversation:
            self.memory.consolidate(
                conversation,
                self.provider,
                self.model,
                source=source,
            )
    
    # --- Private methods ---
    
    def _measure_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Measure total tokens in messages + tool definitions."""
        tool_defs = self.tools.get_definitions()
        tokens, _ = estimate_prompt_tokens_chain(
            self.provider, self.model, messages, tool_defs
        )
        return tokens
    
    def _compact_tool_results(
        self,
        messages: list[dict[str, Any]],
        preserve_last_n: int,
    ) -> None:
        """Compact old tool result content (strategy 1)."""
        # Find tool result messages
        tool_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "tool"
        ]
        
        if len(tool_indices) <= preserve_last_n:
            return
        
        to_compact = tool_indices[:-preserve_last_n]
        compacted_count = 0
        
        for idx in to_compact:
            content = messages[idx].get("content", "")
            if not content.startswith("[compacted"):
                compacted = self._compact_content(content)
                messages[idx] = {**messages[idx], "content": compacted}
                compacted_count += 1
        
        if compacted_count > 0:
            logger.debug("Compacted {} old tool results", compacted_count)
    
    def _compact_content(self, content: str) -> str:
        """Compact tool result content, preserving key information."""
        text = content.strip()
        if not text:
            return self._COMPACTED_PLACEHOLDER
        
        if text.startswith(self._COMPACTED_PLACEHOLDER):
            return text
        
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            lines = [text]
        
        # If short enough, keep as-is
        if len(lines) <= (self._COMPACTED_HEAD_LINES + self._COMPACTED_TAIL_LINES):
            snippet = "\n".join(lines)
        else:
            # Keep head and tail
            snippet = "\n".join([
                *lines[:self._COMPACTED_HEAD_LINES],
                "...",
                *lines[-self._COMPACTED_TAIL_LINES:],
            ])
        
        # Truncate if still too long
        if len(snippet) > self._COMPACTED_MAX_CHARS:
            head = snippet[: int(self._COMPACTED_MAX_CHARS * 0.7)].rstrip()
            tail = snippet[-int(self._COMPACTED_MAX_CHARS * 0.2):].lstrip()
            snippet = f"{head}\n...\n{tail}"
        
        return f"{self._COMPACTED_PLACEHOLDER}\n{snippet}"
    
    def _drop_old_turns(
        self,
        messages: list[dict[str, Any]],
        preserve_last_n: int,
    ) -> None:
        """Drop old assistant+tool turn pairs (strategy 2)."""
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
        dropped_count = len(to_drop)
        
        # Delete front-to-back with offset tracking to handle non-adjacent turns
        offset = 0
        for start, end in to_drop:
            adj_start, adj_end = start - offset, end - offset
            del messages[adj_start:adj_end + 1]
            offset += (end - start + 1)
        
        logger.debug("Dropped {} old turn pairs", dropped_count)
    
    async def _compress_memory(self, messages: list[dict[str, Any]]) -> None:
        """Compress [Past Knowledge] message if too large (strategy 3)."""
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
            "Memory too large: ~{} tokens (max {}). Compressing.",
            estimated_tokens,
            self.budget.memory_max_tokens,
        )
        
        # Call LLM to compress
        compressed = await self._llm_compress_memory(content)
        messages[past_knowledge_idx]["content"] = compressed
    
    async def _llm_compress_memory(self, content: str) -> str:
        """Use LLM to compress memory content."""
        prompt = f"""Compress this agent memory to ~{self.budget.memory_target_tokens} tokens.

Preserve:
- Key facts, decisions, and user preferences
- Recent events and current task state
- Important file names and paths
- Errors and how they were resolved

Remove:
- Redundant details and verbose descriptions
- Old events that are no longer relevant
- Verbatim dialogue (keep summaries instead)

Current memory:
{content}

Return compressed memory, starting with [Past Knowledge]."""
        
        try:
            response = await self.provider.chat_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=self.budget.memory_target_tokens + 200,
            )
            
            compressed = response.content or ""
            if not compressed.startswith("[Past Knowledge]"):
                compressed = f"[Past Knowledge]\n{compressed}"
            
            logger.info(
                "Compressed memory from {} to {} chars",
                len(content),
                len(compressed),
            )
            return compressed
            
        except Exception as e:
            logger.error("Failed to compress memory: {}", e)
            # Fallback: truncate to max tokens
            max_chars = self.budget.memory_max_tokens * 4
            if len(content) > max_chars:
                truncated = content[:max_chars] + "\n\n[Truncated due to size]"
                return truncated
            return content
