clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -f .*.swp
	rm -rf .ropeproject
	rm -f cse415-project.zip

dist:
	zip -r cse415-project . -x *.git*
