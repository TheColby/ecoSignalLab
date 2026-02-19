# Palette's Journal

## 2025-05-15 - [CLI Intent Recognition]
**Learning:** In CLI tools with multiple subcommands (like `analyze` and `batch`), users often confuse the expected input type (file vs directory). Providing a standard error message with a direct suggestion for the alternative command significantly reduces friction.
**Action:** Always check if the input path type matches the command's intent (e.g., file vs directory) and suggest the corresponding subcommand if they don't match.
