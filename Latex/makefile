.RECIPEPREFIX +=
output-directory=temp-dir
output-name=tesi

all:
    latexmk -pdf\
            -output-directory=$(output-directory)\
            -jobname=$(output-name)\
            main.tex
    cp $(output-directory)/$(output-name).pdf .
    rm $(output-directory)/*

continuous:
    latexmk -pdf -pvc -view=none -synctex=1 -interaction=batchmode\
            -output-directory=$(output-directory)\
            -jobname=$(output-name)\
            main.tex

clean:
    rm $(output-directory)/*
