You are doing important research work. This code (and the data it produces) will be seen by other researchers and professors, and if you're lucky, they'll want to run it and build on it. Your code must be easy to maintain and understand. That means that you need to write tests, documentation, and specifications.

### Tests
Your tests should be thorough. Write regression tests that fail *before* fixing a bug.

### Documentation
Your documentation should be thorough, documenting every feature, but it should not be hard to navigate. The project should have a README, as well as each subrepo. Use READMEs as a table of contents, providing a brief summary and directing future devs/users to where they can read a more thorough description. Don't make people read what they don't have to. Do let them read what they need to. Document edge cases and behavior people may find unexpected. Write inline comments for trickier sections.

### Specification
Write specifications for every class, function, method, or their equivalents that you write. This should include representation invariants, inputs, outputs, mutations, etc.

# Writing New Code
When writing new code
1. Write the documentation for the new code but mark it as in progress. Keep documentation for any old code until it is removed. A reader should never be confused about what the current code actually does. This is your planning phase, consult your advisor as needed.
2. Write the specification for the new code.
3. Write the tests for the new code, some of them should fail.
4. Write the new code.
5. Update the documentation with the current state.
6. Commit.
