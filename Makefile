# Makefile

# Target to run the simulation and generate the animation
run:
	python src/main.py

# Target to clean up any generated files (e.g., output video)
clean:
	rm -f output/*.mp4
