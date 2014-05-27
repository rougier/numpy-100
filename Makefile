MAKE             = /usr/bin/make
RST2HTML         = rst2html.py
RST2LATEX        = rst2latex.py
STYLESHEET       = numpy.css
RST2HTML_OPTIONS = --strip-comments             \
                   --report=3                   \
	               --stylesheet=$(STYLESHEET)   \
                   --link-stylesheet
RST2LATEX_OPTIONS = --strip-comments            \
                   --report=3                   \
                   --use-latex-toc

SOURCES = $(wildcard *.rst)
HTML_OBJECTS = $(subst .rst,.html, $(SOURCES))
LATEX_OBJECTS = $(subst .rst,.tex, $(SOURCES))

html: $(HTML_OBJECTS)

latex: $(LATEX_OBJECTS)

%.html: %.rst Makefile
	@echo "  - $@"
	@$(RST2HTML) $(RST2HTML_OPTIONS) $< $@

%.tex: %.rst Makefile
	@echo "  - $@"
	@$(RST2LATEX) $(RST2LATEX_OPTIONS) $< $@

clean:
	@-rm -f $(LATEX_OBJECTS) $(HTML_OBJECTS)

distclean: clean
	@-rm -f `find . -name "*~"`


