Short description of why the PR is needed and how it satisfies those requirements, in sentence form.

(Delete unused sections)
Added public API:
- symbol (e.g. `vcm.my_function`) or script and optional description of changes or why they are needed
- Can group multiple related symbols on a single bullet

Refactored public API:
- Bulleted list of removed or refactored symbols, such as changes to name, type, behavior, argument, etc. Be cautious about doing these and discuss with team more broadly.

Significant internal changes:
- Bulleted list of changes to non-public API

Requirement changes:
- Bulleted list, if relevant, of any changes to setup.py, requirement.txt, environment.yml, etc
- [ ] Ran `make lock_deps/lock_pip` following these [instructions](https://vulcanclimatemodeling.com/docs/fv3net/dependency_management.html#dependency-management)
- [ ] Add PR review with license info for any additions to `constraints.txt`
  ([example](https://github.com/VulcanClimateModeling/fv3net/pull/1218#pullrequestreview-663644359))

- [ ] Tests added

Resolves #<github issues> [JIRA-TAG]

(Delete this for the commit message)
You are encouraged to check the PR against the [Code Review Checklist](https://paper.dropbox.com/doc/Code-Review-Checklist--A4lKrs~xg7w5Gsb39N6JLNQoAg-IlsYffZgTwyKEylty7NhY).
