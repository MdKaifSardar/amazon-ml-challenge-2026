# Agent Operating Rules & Behavioral Guidelines

## 1. Default Mode: Read-Only & Simple Explanation
Unless the user explicitly specifies an action command such as:
- "make changes"
- "make plan"
- "proceed"
- "fix"
- "execute" / "run"
- "update files"

The agent MUST adhere strictly to the following:
1. **No Code or File Modifications**: Never alter any existing code, scripts, configuration, or datasets.
2. **No Workflow Changes**: Do not change the active execution pipeline or model strategy.
3. **No Unsolicited Plans/Artifacts**: Do not generate new plans, roadmaps, or artifacts unless explicitly requested.
4. **Read-Only Analysis**: Purely inspect the existing state, logs, results, or code in read-only mode.
5. **Simple, Plain-English Explanation**: Clearly and simply explain the findings, system state, or concept to the user without overwhelming jargon.
