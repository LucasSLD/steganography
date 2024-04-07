#!/bin/bash

# Define the LaTeX document filename (without extension)
filename="report"
slidename="presentation"

# Remove LaTeX auxiliary files
rm -f "$filename.aux" "$filename.log" "$filename.out" "$filename.toc" "$filename.lof" "$filename.lot" "$filename.bbl" "$filename.blg" "$filename.synctex.gz" "$filename.gz" "$filename.fdb_latexmk" "$filename.fls" "$filename.auxlock" "$filename.run.xml" "$filename.bcf" "$filename.nav" "$filename.snm" "$filename.loa" "$filename.lol" "$filename.tdo" "$filename.bbl-SAVE-ERROR" "$filename.bcf-SAVE-ERROR"

rm -f "$slidename.aux" "$slidename.bcf" "$slidename.fdb_latexmk" "$slidename.fls" "$slidename.log" "$slidename.nav" "$slidename.out" "$slidename.run.xml" "$slidename.snm" "$slidename.synctex.gz" "$slidename.toc" "$slidename.bbl" "$slidename.bbl-SAVE-ERROR" "$slidename.blg"