# Playlist recommendations

"Given this playlist (a folder, or a list of post ids), what else belongs?" Read-only on the
library data; the only write is the optional yes/no feedback.

- API: `POST /recommend/playlist`, `POST /recommend/feedback` ([api/recommend.py](../src/event_driven_rag_service/api/recommend.py))
- UI: `/ui/recommend`
- Maths: [services/playlist_scoring.py](../src/event_driven_rag_service/services/playlist_scoring.py) (pure numpy)
- Flow: [services/playlist_recommendation.py](../src/event_driven_rag_service/services/playlist_recommendation.py)
- SQL: [repository/recommendation_repository.py](../src/event_driven_rag_service/repository/recommendation_repository.py)

## How it ranks

1. **Playlist vectors**: one embedding per post (mean of its chunks), L2-normalised. Default
   embedding is `summary_title`.
2. **Themes**: spherical k-means, k picked by silhouette; falls back to one theme unless the
   separation is real (silhouette >= 0.15). Used for candidate generation and the UI chips only.
3. **Candidates**: one nearest-neighbour query per theme centre (exact scan; pgvector cannot
   index > 2000 dims, and Qwen3-4B is 2560).
4. **Heat = kNN density**: mean cosine similarity to the `knn` (default 3) nearest playlist
   items. Deliberately *not* a real density estimate (KDE/GMM): in hundreds-to-thousands of
   dimensions distances concentrate and bandwidths are meaningless. kNN similarity stays well
   behaved and handles multi-theme playlists without a clustering assumption.
5. **MMR** re-rank (`diversity` 0-1, default 0.3) so results are not near-duplicates.

Not built (deliberately): background-density correction (dividing by corpus density to demote
"hub" stories) and a learned positive-vs-unlabelled classifier. Add them if results feel generic.

## Absence is not a "no"

A story missing from a folder is **unrated, never a negative**. The ranking learns from
positives only. Explicit feedback, filed per playlist (`folder:N`, or a hash of an ad-hoc id
list, so ad-hoc feedback resets if the id list changes):

- 👍 `yes`: joins the playlist's positives on the next run.
- 👎 `no`: hidden from that playlist's results. Not used as a negative in the scoring.
- Table `recommendation_feedback`, created on the first write; reads tolerate it not existing,
  so a read-only database still serves recommendations.

## Measured quality (homelab library, 1747 embedded stories)

Hold out 25% of a folder, recommend from the rest, count held-out stories in the top 30
(550 held-out items over 6 folders):

| Method | recall@30 |
|---|---|
| single centroid, nearest neighbours (the standard baseline) | 0.082 |
| kNN heat, no diversity | 0.093 |
| kNN heat + MMR (diversity 0.3, default) | 0.113 |
| kNN heat + MMR (diversity 0.5) | 0.098 |

Random would be about 0.017. Caveat: this recall is a **lower bound**. A good recommendation
that is simply not in the folder counts as a miss. Use 👍/👎 for the real precision signal.

## Testing

- Unit: `tests/unit/test_playlist_scoring.py`, `test_playlist_recommendation.py`, `tests/unit/api/test_recommend.py`.
- Live and read-only (`default_transaction_read_only=on`), skipped unless the URL is set:

```
RECOMMEND_LIVE_DB_URL=postgresql://rag:rag@192.168.0.234:5432/rag \
RECOMMEND_LIVE_FOLDER=1,2,13 pytest -m live_db tests/integration/test_recommend_live.py -s
```
