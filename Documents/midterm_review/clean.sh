#!/bin/bash

# Define the LaTeX document filename (without extension)
filename="SALAND_slides"

# Remove LaTeX auxiliary files
rm -f "$filename.aux" "$filename.log" "$filename.out" "$filename.toc" "$filename.lof" "$filename.lot" "$filename.bbl" "$filename.blg" "$filename.synctex.gz" "$filename.gz" "$filename.fdb_latexmk" "$filename.fls" "$filename.auxlock" "$filename.run.xml" "$filename.bcf" "$filename.nav" "$filename.snm"
