# Cluster Model Ops Source Workflow

This is the LivingAgile/mlx-lm working fork. Root AGENTS.md contains the owner's
fork-development and safe branch-migration authority. Upstream submissions remain
separate and must follow the upstream target's current instructions.

`cluster-modelops/baseline` is owner policy commit
2f8d889a14c0f6a73e842ff55f817c744575c8c9, based on upstream mirror
a51b9093292cb2a98a6a618d264ddc8948166e6c. Keep main unchanged.

`cluster-modelops/integration` preserves historical integration
7c4a3669d87d0cc9bdc63433030932850221f40e. Its runtime source is unchanged by the
policy transition; it does not include every later model experiment. EXO's existing
pyproject.toml and uv.lock pin that historical MLX-LM revision. Keep the pin until
a separately validated integration effort selects a new dependency revision.

Use `cluster-modelops/feature/<topic>` for development. Inspect outgoing commits,
verify the exact owner push URL, perform applicable checks and fast-forward the
integration branch from its observed tip. Never overwrite a concurrent advance.
Verify the published SHA. Publication is not cluster deployment.

Migration requires a recorded old/new ref and SHA map, preservation of every source
commit, publication verification, and worktree/build/automation/PR dependency checks.
Retain any legacy ref with active work or unresolved consumers. Restore a deleted
owned ref from the recorded SHA only after checking that its name is unoccupied.
Never alter unrelated refs, upstream branches, main, protections or published history.

For upstream publication, branch from an upstream revision and port only the selected
source, tests and reproduction material. Inspect every outgoing commit and the final
diff; exclude fork policy and private ancestry. Never disguise provenance. An export
rehearsal is not a submission or a claim of fresh model qualification.