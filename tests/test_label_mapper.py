from __future__ import annotations

import pandas as pd

from matcher_agent.data.label_mapper import map_status_to_label


def test_map_status_to_label_splits_unknown_values() -> None:
    df = pd.DataFrame(
        [
            {"match_id": "1", "status": "accepted"},
            {"match_id": "4", "status": "closed"},
            {"match_id": "2", "status": "declined"},
            {"match_id": "3", "status": "pending"},
        ]
    )
    accepted, rejects = map_status_to_label(df)
    assert len(accepted) == 3
    assert len(rejects) == 1
    assert set(accepted["label"].tolist()) == {0, 1}
