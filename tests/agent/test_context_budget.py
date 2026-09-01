"""Tests for unified context budget management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context_budget import ContextBudget, ContextBudgetManager
from nanobot.utils.prompt_budget import PromptBudget, reduce_messages_to_budget


class TestContextBudget:
    """Tests for ContextBudget dataclass."""
    
    def test_available_budget_calculation(self) -> None:
        """Test that available_budget = max_tokens - output_reserve - safety_buffer."""
        budget = ContextBudget(
            max_tokens=12000,
            output_reserve=2000,
            safety_buffer=1024,
        )
        
        assert budget.available_budget == 12000 - 2000 - 1024
        assert budget.available_budget == 8976
    
    def test_trigger_tokens_at_100_percent(self) -> None:
        """Test trigger_tokens with trigger_ratio=1.0 (100% budget)."""
        budget = ContextBudget(
            max_tokens=12000,
            output_reserve=2000,
            safety_buffer=1024,
            trigger_ratio=1.0,
        )
        
        assert budget.trigger_tokens == budget.available_budget
        assert budget.trigger_tokens == 8976
    
    def test_trigger_tokens_at_80_percent(self) -> None:
        """Test trigger_tokens with trigger_ratio=0.8 (80% budget)."""
        budget = ContextBudget(
            max_tokens=12000,
            output_reserve=2000,
            safety_buffer=1024,
            trigger_ratio=0.8,
        )
        
        expected = int(8976 * 0.8)
        assert budget.trigger_tokens == expected
    
    def test_target_tokens_at_20_percent(self) -> None:
        """Test target_tokens with target_ratio=0.2 (20% budget)."""
        budget = ContextBudget(
            max_tokens=12000,
            output_reserve=2000,
            safety_buffer=1024,
            target_ratio=0.2,
        )
        
        expected = int(8976 * 0.2)
        assert budget.target_tokens == expected


class TestContextBudgetManager:
    """Tests for ContextBudgetManager class."""
    
    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create a mock LLM provider."""
        provider = MagicMock()
        # Mock estimate_prompt_tokens which is called by estimate_prompt_tokens_chain
        provider.estimate_prompt_tokens.return_value = (100, "mock")
        return provider
    
    @pytest.fixture
    def mock_tools(self) -> MagicMock:
        """Create a mock tool registry."""
        tools = MagicMock()
        tools.get_definitions.return_value = []
        return tools
    
    @pytest.fixture
    def mock_memory(self, tmp_path: Path) -> MagicMock:
        """Create a mock memory store."""
        memory = MagicMock()
        memory.read_long_term.return_value = "Test memory content"
        memory.consolidate = MagicMock()
        return memory
    
    @pytest.fixture
    def budget_manager(
        self,
        mock_provider: MagicMock,
        mock_tools: MagicMock,
        mock_memory: MagicMock,
    ) -> ContextBudgetManager:
        """Create a ContextBudgetManager for testing."""
        budget = ContextBudget(
            max_tokens=1000,
            output_reserve=200,
            safety_buffer=100,
            trigger_ratio=1.0,
            target_ratio=0.2,
        )
        
        return ContextBudgetManager(
            budget=budget,
            memory_store=mock_memory,
            provider=mock_provider,
            model="test-model",
            tool_registry=mock_tools,
        )
    
    async def test_enforce_budget_no_reduction_needed(
        self,
        budget_manager: ContextBudgetManager,
        mock_provider: MagicMock,
    ) -> None:
        """Test that no reduction is applied when under budget."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]
        
        # Mock token count to be under budget
        mock_provider.count_tokens.return_value = 500
        
        result = await budget_manager.enforce_budget(messages)
        
        assert result == messages
        assert len(result) == 2
    
    async def test_enforce_budget_compacts_tool_results(
        self,
        budget_manager: ContextBudgetManager,
        mock_provider: MagicMock,
    ) -> None:
        """Test that old tool results are compacted when over budget."""
        # 3 turns with tool results - with preserve=1, first 2 can be compacted
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Using tool 1", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "Tool result 1 " * 50},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Using tool 2", "tool_calls": [{"id": "2"}]},
            {"role": "tool", "content": "Tool result 2 " * 50},
            {"role": "user", "content": "Question 3"},
            {"role": "assistant", "content": "Using tool 3", "tool_calls": [{"id": "3"}]},
            {"role": "tool", "content": "Tool result 3 " * 50},
        ]
        
        # Mock: over budget initially, then under budget after compaction
        # Key: return under target (140) so no drops happen and compacted messages survive
        mock_provider.estimate_prompt_tokens.side_effect = [
            (800, "mock"),  # Initial measurement (over budget of 700)
            (100, "mock"),  # After compaction (under target of 140, so no drops)
        ]
        
        result = await budget_manager.enforce_budget(messages, preserve_last_n_turns=1)
        
        # Should have compacted at least one tool result
        # With no drops (because we're under target after compaction), compacted messages survive
        tool_messages = [m for m in result if m.get("role") == "tool"]
        compacted_count = sum(1 for m in tool_messages if "[compacted" in str(m.get("content", "")))
        assert compacted_count >= 1, f"Expected at least one compacted tool result, got {compacted_count} out of {len(tool_messages)}"
    
    async def test_enforce_budget_drops_old_turns(
        self,
        budget_manager: ContextBudgetManager,
        mock_provider: MagicMock,
    ) -> None:
        """Test that old turn pairs are dropped when compaction is insufficient."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
            {"role": "user", "content": "Third question"},
            {"role": "assistant", "content": "Third answer"},
        ]
        
        # Mock: over budget, compact doesn't help enough, then drop helps
        mock_provider.estimate_prompt_tokens.side_effect = [
            (800, "mock"),  # Initial (over budget)
            (800, "mock"),  # After compact (still over, no tool results to compact)
            (200, "mock"),  # After drop (under target)
        ]
        
        initial_len = len(messages)
        result = await budget_manager.enforce_budget(messages, preserve_last_n_turns=1)
        
        # Should have dropped some old turns (messages list mutated in place)
        # Note: enforce_budget returns the same list, mutated
        assert len(result) < initial_len, f"Expected fewer messages after drop, got {len(result)} vs {initial_len}"
    
    def test_compact_content_preserves_short_content(
        self,
        budget_manager: ContextBudgetManager,
    ) -> None:
        """Test that short content is not compacted."""
        short_content = "This is a short tool result."
        
        result = budget_manager._compact_content(short_content)
        
        # Should start with compacted placeholder but preserve content
        assert "[compacted" in result
        assert "short tool result" in result
    
    def test_compact_content_truncates_long_content(
        self,
        budget_manager: ContextBudgetManager,
    ) -> None:
        """Test that long content is truncated."""
        long_content = "Line\n" * 100
        
        result = budget_manager._compact_content(long_content)
        
        # Should be compacted and truncated
        assert "[compacted" in result
        assert len(result) < len(long_content)
        assert "..." in result  # Truncation marker
    
    def test_compact_content_idempotent(
        self,
        budget_manager: ContextBudgetManager,
    ) -> None:
        """Test that compacting already-compacted content is idempotent."""
        original = "Some content"
        compacted_once = budget_manager._compact_content(original)
        compacted_twice = budget_manager._compact_content(compacted_once)
        
        # Should not double-compact
        assert compacted_once == compacted_twice

    def test_compact_content_handles_mixed_multimodal_blocks_without_payloads(
        self,
        budget_manager: ContextBudgetManager,
    ) -> None:
        encoded = "A" * 512
        content = [
            {"type": "text", "text": "Useful text from the tool."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
            {"type": "future_media", "payload": encoded},
        ]

        compacted = budget_manager._compact_content(content)

        assert "Useful text from the tool." in compacted
        assert "[image" in compacted
        assert "[audio" in compacted
        assert "future_media" in compacted
        assert encoded not in compacted
        assert "base64" not in compacted

    async def test_enforce_budget_handles_mixed_multimodal_tool_results(
        self,
        budget_manager: ContextBudgetManager,
        mock_provider: MagicMock,
    ) -> None:
        encoded = "Q" * 512
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Using visual tool", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": [
                {"type": "text", "text": "Keep this finding."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                {"type": "input_audio", "input_audio": {"data": encoded}},
            ]},
            {"role": "assistant", "content": "Using another tool", "tool_calls": [{"id": "2"}]},
            {"role": "tool", "content": "latest result"},
        ]
        mock_provider.estimate_prompt_tokens.side_effect = [
            (800, "mock"),
            (100, "mock"),
        ]

        await budget_manager.enforce_budget(messages, preserve_last_n_turns=1)

        compacted = messages[2]["content"]
        assert isinstance(compacted, str)
        assert "Keep this finding." in compacted
        assert encoded not in compacted


class TestContextBuilderIntegration:
    """Integration tests for ContextBuilder with inject_memory flag."""
    
    def test_build_active_context_message_empty(self) -> None:
        """Test build_active_context_message with empty memory."""
        from nanobot.agent.context import ContextBuilder
        
        # Create a minimal builder
        builder = object.__new__(ContextBuilder)
        
        result = builder.build_active_context_message("")
        
        assert result["role"] == "user"
        assert "[Past Knowledge]" in result["content"]
        assert "No prior knowledge" in result["content"]
    
    def test_build_active_context_message_with_content(self) -> None:
        """Test build_active_context_message with memory content."""
        from nanobot.agent.context import ContextBuilder
        
        builder = object.__new__(ContextBuilder)
        
        memory_content = "Important fact: user prefers dark mode."
        result = builder.build_active_context_message(memory_content)
        
        assert result["role"] == "user"
        assert "[Past Knowledge]" in result["content"]
        assert "dark mode" in result["content"]


class TestMemoryStoreCompression:
    """Tests for MemoryStore compression methods."""
    
    async def test_compress_to_target_no_compression_needed(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that no compression occurs when under target."""
        from nanobot.agent.memory import MemoryStore
        
        store = MemoryStore(tmp_path)
        short_memory = "Short memory content."
        store.write_long_term(short_memory)
        
        provider = MagicMock()
        
        result = await store.compress_to_target(
            target_tokens=1000,  # Much larger than content
            provider=provider,
            model="test-model",
        )
        
        # Should return content unchanged (no LLM call)
        assert result == short_memory
        provider.chat_with_retry.assert_not_called()
    
    async def test_compress_to_target_calls_llm(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that LLM is called when compression is needed."""
        from nanobot.agent.memory import MemoryStore
        
        store = MemoryStore(tmp_path)
        long_memory = "Word " * 2000  # ~2000 tokens
        store.write_long_term(long_memory)
        
        provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Compressed memory."
        provider.chat_with_retry = AsyncMock(return_value=mock_response)
        
        result = await store.compress_to_target(
            target_tokens=500,  # Much smaller than content
            provider=provider,
            model="test-model",
        )
        
        # Should call LLM for compression
        provider.chat_with_retry.assert_called_once()
        assert "Compressed" in result
    
    async def test_compress_to_target_empty_memory(
        self,
        tmp_path: Path,
    ) -> None:
        """Test compression with empty memory."""
        from nanobot.agent.memory import MemoryStore
        
        store = MemoryStore(tmp_path)
        # Don't write anything to memory
        
        provider = MagicMock()
        
        result = await store.compress_to_target(
            target_tokens=500,
            provider=provider,
            model="test-model",
        )
        
        # Should return empty string
        assert result == ""
        provider.chat_with_retry.assert_not_called()


class TestAppendHistoryWithSource:
    """Tests for append_history with source parameter."""
    
    def test_append_history_with_source_tag(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that source tag is added to history entry."""
        from nanobot.agent.memory import MemoryStore
        
        store = MemoryStore(tmp_path)
        
        store.append_history(
            "[2025-01-15 14:30] Test entry",
            source="cron",
        )
        
        content = store.history_file.read_text()
        assert "[source=cron]" in content
        assert "Test entry" in content
    
    def test_append_history_without_source_tag(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that entry without source is unchanged."""
        from nanobot.agent.memory import MemoryStore
        
        store = MemoryStore(tmp_path)
        
        store.append_history("[2025-01-15 14:30] Test entry")
        
        content = store.history_file.read_text()
        assert "[source=" not in content
        assert "Test entry" in content
    
    def test_append_history_no_double_source_tag(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that source tag is not added if already present."""
        from nanobot.agent.memory import MemoryStore
        
        store = MemoryStore(tmp_path)
        
        store.append_history(
            "[2025-01-15 14:30] [source=chat] Already has source",
            source="cron",  # Different source
        )
        
        content = store.history_file.read_text()
        assert "[source=chat]" in content
        assert "[source=cron]" not in content  # Should not add second source


def test_prompt_reducer_preserves_current_user_and_atomic_newest_tool_group() -> None:
    """A large current turn drops old groups without orphaning tool results."""

    class _LengthProvider:
        @staticmethod
        def estimate_prompt_tokens(messages, _tools, _model):
            content_chars = sum(len(str(message.get("content") or "")) for message in messages)
            return content_chars + len(messages) * 4, "test"

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "old-call"}],
        },
        {"role": "tool", "tool_call_id": "old-call", "content": "x" * 100},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "new-call"}],
        },
        {"role": "tool", "tool_call_id": "new-call", "content": "y" * 20},
    ]

    reduced = reduce_messages_to_budget(
        messages,
        _LengthProvider(),
        "test-model",
        [],
        PromptBudget(total_tokens=100, safety_buffer=0),
    )

    assert reduced[0]["role"] == "system"
    assert reduced[1] == messages[1]
    assert not any(message.get("tool_call_id") == "old-call" for message in reduced)
    assert any(message.get("tool_call_id") == "new-call" for message in reduced)
