# Contributing to MacroEconometricModels.jl

Thank you for your interest in contributing. This guide explains how to report bugs, request features, and submit pull requests.

## Reporting Bugs

Open an [issue](https://github.com/chung9207/MacroEconometricModels.jl/issues) with:

- A clear, descriptive title
- Minimal reproducible example (MWE)
- Expected vs. actual behavior
- Julia version (`versioninfo()`) and package version (`Pkg.status("MacroEconometricModels")`)

## Requesting New Features or Models

Open an [issue](https://github.com/chung9207/MacroEconometricModels.jl/issues) with:

- A description of the feature or econometric model
- Key references (papers, textbooks) for new models
- A brief explanation of why it would be useful

## Pull Requests

Pull requests are welcome for bug fixes and new models/features. Before starting significant work, please open an issue first to discuss the approach.

### Setup

```bash
git clone https://github.com/chung9207/MacroEconometricModels.jl.git
cd MacroEconometricModels.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Workflow

1. Fork the repository and create a branch from `main`
2. Make your changes
3. Add tests in the appropriate `test/` subdirectory (mirror the `src/` structure)
4. Run the test suite and confirm all tests pass:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```
5. Submit a pull request against `main`

### Guidelines

- Follow existing code conventions: `T<:AbstractFloat` type parameters, `robust_inv` over raw `inv`, `_` prefix for internal helpers
- Use `Optim.optimize` / `Optim.LBFGS()` (qualified calls, not bare names)
- Add docstrings for all public functions
- New models should include: types in `types.jl`, estimation, analysis functions, tests, and a documentation page
- Keep PRs focused — one feature or fix per PR

### Running Documentation Locally

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Code of Conduct

Be respectful and constructive. We are all here to build good econometric tools.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
