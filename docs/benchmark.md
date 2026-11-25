# CrossTrace benchmark

## Task

One query is one downstream issue. The system returns a ranked list of
upstream suspects. The gold answers are the upstream items that a human
or the AutoAnnotator labeled `cause`.

## Items

- A query has: ID, repository, title, body, date, and the gold IDs.
- A candidate has: ID, repository, kind (issue, PR, release), title, body,
  and date.
- An ID looks like `numpy/numpy#123`.
- The date is the creation date. For a release it is the publish date.
- Every date has a time zone.

## Rules

1. **Time rule.** A candidate must be strictly older than the query.
   An item with the same time as the query is not allowed.
2. **Same repository.** By default, a candidate from the repository of the
   query is not allowed. The task is cross-project.
3. **Cleaning.** A gold answer is removed if it is not in the candidate pool
   or if it breaks rule 1 or 2. A query with no gold answer is removed.
   The cleaning report counts what was removed.

## Splits

- **Time split.** The newest 20% of the queries are the test set.
  The cut is at a date. Queries with the same date stay together.
- **Leave-one-project-out.** One split for each repository.
  The test set holds the queries of that repository.
  The training set holds the queries of all other repositories.

## Known limits

- In leave-one-project-out, a training query can have a gold answer from the
  test repository. The test repository is not a query source in training.
  It can still appear as a gold answer. The metrics are not wrong, but the
  model can learn about that repository.
- The time split does not remove old candidates from the pool. A test query
  can use any candidate that is older than the query. This is correct,
  because the information was available at that time.
- A `cause` label does not say which item is upstream. Building queries needs
  this direction. It is not solved in this document.