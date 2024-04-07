echo "Compiling LaTeX document..."
pdflatex presentation.tex

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
./clean.sh

echo "Compilation complete."