"""
Lightweight progress tracking for psych norms generation.

Strategy: Track completion at the AGGREGATE level per (model, temperature).
For detailed per-cue continuation, rely on local CSV files on the cluster.

The manifest is ultra-compact (~1KB) and only tracks:
- Which (model, temp) combinations are FULLY complete
- Partial progress counts for in-progress runs
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Tuple


class ProgressManifest:
    """
    Ultra-compact manifest tracking aggregate completion status.
    
    Structure:
    {
        "version": 3,
        "complete": ["model|temp", ...],  # Fully completed task groups
        "progress": {
            "model|temp": {
                "total_tasks": int,  # Total unique (word, norm) pairs
                "completed": int,    # Count of completed ratings
                "target_reps": int   # Target repetitions (1 or 5)
            }
        }
    }
    
    This provides O(1) lookup to skip fully completed models,
    while detailed continuation within a run uses cluster-local CSVs.
    """
    
    VERSION = 3
    
    def __init__(self, path: str):
        self.path = Path(path)
        self.data = self._load()
    
    def _load(self) -> dict:
        """Load manifest from disk or return empty structure."""
        if self.path.exists():
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("version") == self.VERSION:
                        return data
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Could not load manifest: {e}")
        
        return {"version": self.VERSION, "complete": [], "progress": {}}
    
    def save(self) -> None:
        """Save manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write
        fd, temp_path = tempfile.mkstemp(dir=self.path.parent, suffix='.json')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            os.replace(temp_path, self.path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def _task_key(self, model: str, temperature: float) -> str:
        return f"{model}|{temperature:.1f}"
    
    def is_complete(self, model: str, temperature: float) -> bool:
        """Check if a model/temp combination is fully complete."""
        return self._task_key(model, temperature) in self.data["complete"]
    
    def get_progress(self, model: str, temperature: float) -> dict:
        """Get progress info for a model/temp combination."""
        key = self._task_key(model, temperature)
        return self.data["progress"].get(key, {
            "total_tasks": 0,
            "completed": 0,
            "target_reps": 1
        })
    
    def mark_complete(self, model: str, temperature: float) -> None:
        """Mark a model/temp combination as fully complete."""
        key = self._task_key(model, temperature)
        if key not in self.data["complete"]:
            self.data["complete"].append(key)
        # Remove from progress since it's complete
        if key in self.data["progress"]:
            del self.data["progress"][key]
    
    def update_progress(self, model: str, temperature: float, 
                        total_tasks: int, completed: int, target_reps: int) -> None:
        """Update progress for a model/temp combination."""
        key = self._task_key(model, temperature)
        
        # Check if fully complete
        expected_completions = total_tasks * target_reps
        if completed >= expected_completions:
            self.mark_complete(model, temperature)
        else:
            self.data["progress"][key] = {
                "total_tasks": total_tasks,
                "completed": completed,
                "target_reps": target_reps
            }
    
    def should_skip(self, model: str, temperature: float) -> bool:
        """
        Determine if a model/temp run should be skipped entirely.
        Returns True only if marked as fully complete.
        """
        return self.is_complete(model, temperature)
    
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = ["Progress Manifest Summary:"]
        lines.append(f"  Complete: {len(self.data['complete'])} task groups")
        for key in self.data['complete']:
            lines.append(f"    ✓ {key}")
        
        lines.append(f"  In Progress: {len(self.data['progress'])} task groups")
        for key, info in self.data['progress'].items():
            pct = (info['completed'] / (info['total_tasks'] * info['target_reps']) * 100) if info['total_tasks'] > 0 else 0
            lines.append(f"    ⏳ {key}: {info['completed']:,}/{info['total_tasks'] * info['target_reps']:,} ({pct:.1f}%)")
        
        return "\n".join(lines)
