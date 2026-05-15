# logs/

This directory will hold Harbor run output once `./run_eval.sh` is executed
against the live `gemini-3-flash-preview` model.

For each task, three trial logs are produced:

```
<task-name>.gemini.trial1.log
<task-name>.gemini.trial2.log
<task-name>.gemini.trial3.log
```

Optional oracle / nop sanity logs are produced by `./run_eval.sh sanity`:

```
<task-name>.oracle.log
<task-name>.nop.log
```

The runner pipes both the agent transcript and the verifier reward to the
log file. The final reward (0 or 1) is written by the verifier to
`/logs/verifier/reward.txt` inside the container and echoed in the log.
