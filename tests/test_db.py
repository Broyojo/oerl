"""
Comprehensive tests for the VectorDatabase class.
Tests all functionality including embedding, adding, and searching trajectories.
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
from grader.db import VectorDatabase


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_db(temp_db_path):
    """Create a VectorDatabase instance with a temporary path."""
    return VectorDatabase(
        db_path=temp_db_path,
        collection_name="test_trajectories",
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


class TestVectorDatabaseInitialization:
    """Test database initialization and setup."""

    def test_init_creates_database(self, temp_db_path):
        """Test that initialization creates a database instance."""
        db = VectorDatabase(
            db_path=temp_db_path,
            collection_name="test_collection",
        )

        assert db is not None
        assert db.collection is not None
        assert db.text_model is not None
        assert db.text_dim > 0

    def test_init_uses_default_parameters(self, temp_db_path):
        """Test that default parameters are applied correctly."""
        db = VectorDatabase(db_path=temp_db_path)

        assert db.collection.name == "trajectories"
        assert db.text_dim == 384  # all-MiniLM-L6-v2 has 384 dimensions

    def test_init_custom_collection_name(self, temp_db_path):
        """Test that custom collection names are respected."""
        custom_name = "my_custom_collection"
        db = VectorDatabase(
            db_path=temp_db_path,
            collection_name=custom_name,
        )

        assert db.collection.name == custom_name

    def test_persistent_client_reuses_existing_collection(self, temp_db_path):
        """Test that reopening a database reuses existing collections."""
        # Create first instance
        db1 = VectorDatabase(
            db_path=temp_db_path,
            collection_name="persistent_test",
        )
        db1.add(
            trajectory_id="test-persist-1",
            trajectory="Test trajectory for persistence",
            metadata={"test": True},
        )

        # Create second instance with same path and collection
        db2 = VectorDatabase(
            db_path=temp_db_path,
            collection_name="persistent_test",
        )

        # Search should find the previously added trajectory
        results = db2.search(
            trajectory="Test trajectory for persistence",
            n_results=1,
        )

        assert len(results["ids"][0]) > 0
        assert "test-persist-1" in results["ids"][0]


class TestVectorDatabaseEmbedding:
    """Test embedding functionality."""

    def test_embed_returns_list_of_floats(self, vector_db):
        """Test that _embed returns a list of floats."""
        text = "This is a test trajectory"
        embedding = vector_db._embed(text)

        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        assert len(embedding) == vector_db.text_dim

    def test_embed_none_returns_zero_vector(self, vector_db):
        """Test that embedding None returns a zero vector."""
        embedding = vector_db._embed(None)

        assert isinstance(embedding, list)
        assert len(embedding) == vector_db.text_dim
        assert all(x == 0.0 for x in embedding)

    def test_embed_empty_string(self, vector_db):
        """Test that embedding empty string works."""
        embedding = vector_db._embed("")

        assert isinstance(embedding, list)
        assert len(embedding) == vector_db.text_dim
        # Empty string may produce near-zero or all-zero embeddings depending on model
        # Just verify it returns a valid embedding vector
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_produces_normalized_vectors(self, vector_db):
        """Test that embeddings are normalized (norm close to 1)."""
        text = "Normalized embedding test"
        embedding = vector_db._embed(text)

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be close to 1

    def test_embed_similar_texts_produce_similar_embeddings(self, vector_db):
        """Test that similar texts produce similar embeddings."""
        text1 = "The cat sat on the mat"
        text2 = "A cat was sitting on a mat"
        text3 = "Quantum physics is fascinating"

        emb1 = np.array(vector_db._embed(text1))
        emb2 = np.array(vector_db._embed(text2))
        emb3 = np.array(vector_db._embed(text3))

        # Similar texts should have higher cosine similarity
        similarity_12 = np.dot(emb1, emb2)
        similarity_13 = np.dot(emb1, emb3)

        assert similarity_12 > similarity_13


class TestVectorDatabaseAdd:
    """Test adding trajectories to the database."""

    def test_add_single_trajectory(self, vector_db):
        """Test adding a single trajectory."""
        vector_db.add(
            trajectory_id="traj-001",
            trajectory="User asks about AI. Agent responds with detailed explanation.",
            metadata={"score": 85.5, "category": "AI"},
        )

        # Verify it was added by searching for it
        results = vector_db.search(
            trajectory="AI explanation",
            n_results=1,
        )

        assert len(results["ids"][0]) > 0
        assert "traj-001" in results["ids"][0]

    def test_add_includes_trajectory_text_in_metadata(self, vector_db):
        """Test that the trajectory text is stored in metadata."""
        trajectory_text = "This is the trajectory content"
        vector_db.add(
            trajectory_id="traj-002",
            trajectory=trajectory_text,
            metadata={"custom": "value"},
        )

        results = vector_db.search(
            trajectory=trajectory_text,
            n_results=1,
        )

        assert results["metadatas"][0][0]["trajectory_text"] == trajectory_text
        assert results["metadatas"][0][0]["custom"] == "value"

    def test_add_multiple_trajectories(self, vector_db):
        """Test adding multiple trajectories."""
        trajectories = [
            ("traj-100", "First trajectory about coding", {"topic": "code"}),
            ("traj-101", "Second trajectory about art", {"topic": "art"}),
            ("traj-102", "Third trajectory about science", {"topic": "science"}),
        ]

        for traj_id, traj_text, metadata in trajectories:
            vector_db.add(
                trajectory_id=traj_id,
                trajectory=traj_text,
                metadata=metadata,
            )

        # Search for each and verify
        for traj_id, traj_text, metadata in trajectories:
            results = vector_db.search(trajectory=traj_text, n_results=1)
            assert traj_id in results["ids"][0]

    def test_upsert_overwrites_existing_trajectory(self, vector_db):
        """Test that adding with same ID updates the trajectory."""
        traj_id = "traj-update"

        # Add initial version
        vector_db.add(
            trajectory_id=traj_id,
            trajectory="Original content",
            metadata={"version": 1},
        )

        # Update with new content
        vector_db.add(
            trajectory_id=traj_id,
            trajectory="Updated content",
            metadata={"version": 2},
        )

        # Search should return the updated version
        results = vector_db.search(trajectory="Updated content", n_results=1)

        assert traj_id in results["ids"][0]
        assert results["metadatas"][0][0]["version"] == 2
        assert "Updated content" in results["metadatas"][0][0]["trajectory_text"]

    def test_add_with_empty_metadata(self, vector_db):
        """Test adding trajectory with empty metadata."""
        vector_db.add(
            trajectory_id="traj-empty-meta",
            trajectory="Trajectory with no extra metadata",
            metadata={},
        )

        results = vector_db.search(
            trajectory="Trajectory with no extra metadata",
            n_results=1,
        )

        assert "traj-empty-meta" in results["ids"][0]
        # Should still have trajectory_text in metadata
        assert "trajectory_text" in results["metadatas"][0][0]


class TestVectorDatabaseSearch:
    """Test searching for similar trajectories."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, vector_db):
        """Add test trajectories before each test."""
        test_trajectories = [
            (
                "traj-search-1",
                "Agent explores creative visual programming with colors and shapes",
                {"score": 90.0, "category": "creative"},
            ),
            (
                "traj-search-2",
                "Agent performs simple arithmetic calculation",
                {"score": 30.0, "category": "mundane"},
            ),
            (
                "traj-search-3",
                "Agent investigates quantum entanglement and consciousness",
                {"score": 95.0, "category": "deep"},
            ),
            (
                "traj-search-4",
                "Agent creates algorithmic music from chaos theory",
                {"score": 88.0, "category": "artistic"},
            ),
            (
                "traj-search-5",
                "Agent checks the weather forecast",
                {"score": 25.0, "category": "standard"},
            ),
        ]

        for traj_id, trajectory, metadata in test_trajectories:
            vector_db.add(
                trajectory_id=traj_id,
                trajectory=trajectory,
                metadata=metadata,
            )

        return test_trajectories

    def test_search_returns_similar_trajectories(self, vector_db):
        """Test that search returns semantically similar trajectories."""
        results = vector_db.search(
            trajectory="Agent creates art using programming",
            n_results=3,
        )

        # Should find the creative and artistic trajectories
        found_ids = results["ids"][0]
        assert len(found_ids) > 0
        assert any("search-1" in id or "search-4" in id for id in found_ids)

    def test_search_n_results_limits_returns(self, vector_db):
        """Test that n_results parameter limits the number of results."""
        results = vector_db.search(
            trajectory="Agent does something",
            n_results=2,
        )

        assert len(results["ids"][0]) <= 2

    def test_search_returns_distances(self, vector_db):
        """Test that search returns distance metrics."""
        results = vector_db.search(
            trajectory="quantum physics exploration",
            n_results=3,
        )

        assert "distances" in results
        assert len(results["distances"][0]) > 0
        # Distances should be non-negative
        assert all(d >= 0 for d in results["distances"][0])

    def test_search_returns_metadata(self, vector_db):
        """Test that search returns metadata for each result."""
        results = vector_db.search(
            trajectory="creative exploration",
            n_results=2,
        )

        assert "metadatas" in results
        assert len(results["metadatas"][0]) > 0
        # Each metadata should contain trajectory_text
        for meta in results["metadatas"][0]:
            assert "trajectory_text" in meta

    def test_search_with_where_filter(self, vector_db):
        """Test searching with metadata filter."""
        results = vector_db.search(
            trajectory="Agent explores",
            n_results=5,
            where_filter={"category": "creative"},
        )

        # Should only return creative trajectories
        for meta in results["metadatas"][0]:
            assert meta["category"] == "creative"

    def test_search_most_similar_has_smallest_distance(self, vector_db):
        """Test that results are ordered by similarity (smallest distance first)."""
        # Search for exact match
        results = vector_db.search(
            trajectory="Agent checks the weather forecast",
            n_results=5,
        )

        distances = results["distances"][0]
        # Distances should be in ascending order
        assert distances == sorted(distances)
        # First result should be the exact match with very small distance
        assert distances[0] < 0.5

    def test_search_empty_database(self, temp_db_path):
        """Test searching an empty database."""
        empty_db = VectorDatabase(
            db_path=temp_db_path,
            collection_name="empty_collection",
        )

        results = empty_db.search(
            trajectory="Something",
            n_results=5,
        )

        # Should return empty results
        assert len(results["ids"][0]) == 0

    def test_search_more_results_than_available(self, vector_db):
        """Test requesting more results than exist in database."""
        results = vector_db.search(
            trajectory="test query",
            n_results=100,  # More than the 5 test trajectories
        )

        # Should return all available results (5 from setup)
        assert len(results["ids"][0]) == 5


class TestVectorDatabaseIntegration:
    """Integration tests combining multiple operations."""

    def test_add_search_workflow(self, vector_db):
        """Test complete workflow of adding and searching trajectories."""
        # Add multiple related trajectories
        vector_db.add(
            trajectory_id="int-1",
            trajectory="Exploring machine learning algorithms",
            metadata={"domain": "AI"},
        )
        vector_db.add(
            trajectory_id="int-2",
            trajectory="Understanding neural network architectures",
            metadata={"domain": "AI"},
        )
        vector_db.add(
            trajectory_id="int-3",
            trajectory="Cooking recipes for pasta",
            metadata={"domain": "cooking"},
        )

        # Search for AI-related content
        results = vector_db.search(
            trajectory="artificial intelligence and deep learning",
            n_results=3,
        )

        # Top results should be AI-related
        top_2_ids = results["ids"][0][:2]
        assert "int-1" in top_2_ids or "int-2" in top_2_ids

    def test_read_augment_write_cycle(self, vector_db):
        """Test the read-augment-write cycle used in grading."""
        # Initial write
        vector_db.add(
            trajectory_id="cycle-1",
            trajectory="Creative exploration of alien biology",
            metadata={"score": 85.0},
        )

        # Read (search for similar)
        new_trajectory = "Investigating extraterrestrial life forms"
        similar = vector_db.search(trajectory=new_trajectory, n_results=1)

        assert len(similar["ids"][0]) > 0

        # Write new trajectory (augmented with context from search)
        vector_db.add(
            trajectory_id="cycle-2",
            trajectory=new_trajectory,
            metadata={"score": 82.0, "similar_to": similar["ids"][0][0]},
        )

        # Verify both exist
        results = vector_db.search(trajectory="alien life", n_results=2)
        assert len(results["ids"][0]) == 2

    def test_concurrent_operations(self, vector_db):
        """Test that database handles multiple operations correctly."""
        # Add multiple items
        for i in range(10):
            vector_db.add(
                trajectory_id=f"concurrent-{i}",
                trajectory=f"Test trajectory number {i}",
                metadata={"index": i},
            )

        # Perform multiple searches
        for i in range(5):
            results = vector_db.search(
                trajectory=f"Test trajectory number {i}",
                n_results=3,
            )
            assert len(results["ids"][0]) > 0


class TestVectorDatabaseEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_trajectory_text(self, vector_db):
        """Test adding very long trajectory text."""
        long_text = "A" * 10000  # 10K characters
        vector_db.add(
            trajectory_id="long-traj",
            trajectory=long_text,
            metadata={},
        )

        results = vector_db.search(trajectory=long_text[:100], n_results=1)
        assert "long-traj" in results["ids"][0]

    def test_special_characters_in_trajectory(self, vector_db):
        """Test handling of special characters."""
        special_text = "Test with special chars: @#$%^&*() []{}|\\<>?/~`"
        vector_db.add(
            trajectory_id="special-chars",
            trajectory=special_text,
            metadata={},
        )

        results = vector_db.search(trajectory=special_text, n_results=1)
        assert "special-chars" in results["ids"][0]

    def test_unicode_in_trajectory(self, vector_db):
        """Test handling of unicode characters."""
        unicode_text = "Testing unicode: 你好世界 🌍 émojis 🚀 and symbols ∑∏∫"
        vector_db.add(
            trajectory_id="unicode-traj",
            trajectory=unicode_text,
            metadata={"language": "mixed"},
        )

        results = vector_db.search(trajectory=unicode_text, n_results=1)
        assert "unicode-traj" in results["ids"][0]

    def test_numeric_metadata_values(self, vector_db):
        """Test that numeric metadata values are preserved."""
        vector_db.add(
            trajectory_id="numeric-meta",
            trajectory="Test trajectory",
            metadata={
                "score": 75.5,
                "count": 42,
                "flag": True,
            },
        )

        results = vector_db.search(trajectory="Test trajectory", n_results=1)
        meta = results["metadatas"][0][0]

        assert isinstance(meta["score"], (int, float))
        assert isinstance(meta["count"], int)
        assert isinstance(meta["flag"], bool)
