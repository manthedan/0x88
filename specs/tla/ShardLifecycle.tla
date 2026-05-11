---- MODULE ShardLifecycle ----
EXTENDS FiniteSets

(***************************************************************************)
(* AWS Batch/S3 self-play shard lifecycle model.                            *)
(*                                                                         *)
(* Intent: prove/control-plane-check that a run cannot be finalized unless  *)
(* every expected shard has reached a validated state, and that repairs do  *)
(* not overwrite already committed outputs.                                *)
(***************************************************************************)

CONSTANTS Shards

VARIABLES shardState, committed, committedUris

States == {"pending", "running", "uploaded", "validated", "failed", "repairing"}

vars == << shardState, committed, committedUris >>

Init ==
  /\ shardState = [s \in Shards |-> "pending"]
  /\ committed = FALSE
  /\ committedUris = {}

Start(s) ==
  /\ shardState[s] = "pending"
  /\ shardState' = [shardState EXCEPT ![s] = "running"]
  /\ UNCHANGED << committed, committedUris >>

Upload(s) ==
  /\ shardState[s] = "running"
  /\ shardState' = [shardState EXCEPT ![s] = "uploaded"]
  /\ UNCHANGED << committed, committedUris >>

Validate(s) ==
  /\ shardState[s] = "uploaded"
  /\ shardState' = [shardState EXCEPT ![s] = "validated"]
  /\ UNCHANGED << committed, committedUris >>

Fail(s) ==
  /\ shardState[s] \in {"pending", "running", "uploaded", "repairing"}
  /\ shardState' = [shardState EXCEPT ![s] = "failed"]
  /\ UNCHANGED << committed, committedUris >>

Repair(s) ==
  /\ shardState[s] = "failed"
  /\ shardState' = [shardState EXCEPT ![s] = "repairing"]
  /\ UNCHANGED << committed, committedUris >>

RepairUpload(s) ==
  /\ shardState[s] = "repairing"
  /\ shardState' = [shardState EXCEPT ![s] = "uploaded"]
  /\ UNCHANGED << committed, committedUris >>

CommitRun ==
  /\ committed = FALSE
  /\ \A s \in Shards: shardState[s] = "validated"
  /\ committed' = TRUE
  /\ committedUris' = Shards
  /\ UNCHANGED shardState

Next ==
  \/ \E s \in Shards: Start(s)
  \/ \E s \in Shards: Upload(s)
  \/ \E s \in Shards: Validate(s)
  \/ \E s \in Shards: Fail(s)
  \/ \E s \in Shards: Repair(s)
  \/ \E s \in Shards: RepairUpload(s)
  \/ CommitRun

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ shardState \in [Shards -> States]
  /\ committed \in BOOLEAN
  /\ committedUris \subseteq Shards

NoFinalizeBeforeAllValidated ==
  committed => \A s \in Shards: shardState[s] = "validated"

NoOverwriteCommittedShard ==
  committed => committedUris = Shards

THEOREM Spec => []TypeOK
THEOREM Spec => []NoFinalizeBeforeAllValidated
THEOREM Spec => []NoOverwriteCommittedShard

====
