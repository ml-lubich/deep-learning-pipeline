from dl_pipeline.pipeline.runner import run_batch_pipeline


def test_batches_split_evenly() -> None:
    result = run_batch_pipeline(rows=range(10), batch_size=4)
    assert result.num_batches_published == 3
    assert [len(b) for b in result.batches_received] == [4, 4, 2]


def test_single_batch_when_smaller_than_batch_size() -> None:
    result = run_batch_pipeline(rows=[1, 2], batch_size=10)
    assert result.num_batches_published == 1
    assert result.batches_received == ((1, 2),)
