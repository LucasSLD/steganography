echo "Compiling LaTeX document..."
pdflatex report.tex

# Run Biber
echo "Running Biber..."
biber report

# Compile LaTeX document again
echo "Compiling LaTeX document again..."
pdflatex report.tex

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
./clean.sh

echo "Compilation complete."