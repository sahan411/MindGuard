$ErrorActionPreference = "Stop"

# Build the report PDF using latexmk (Perl-based) and MiKTeX.
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error -outdir=report report/main.tex

if ($LASTEXITCODE -ne 0) {
    throw "Report build failed. Check LaTeX errors above."
}

Write-Output "Report built successfully: report/main.pdf"
