# Contributing

Thanks for your interest in contributing! Whether it's fixing a bug, improving documentation, or proposing new features, your help is appreciated. To contribute:

1. **Fork** this repository to your own GitHub account (instructions [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)).
2. **Clone** your fork to your local machine:
   ```bash
   git clone https://github.com/your-username/aqua-blue.git
   cd aqua-blue/
   ```
3. Create a virtual environment in the `git` clone and activate it:
   ```bash
   python -m venv venv
   
   # on linux/macOS
   source venv/bin/activate
   
   # on Windows
   .\venv\Scripts\activate
   ```
4. Build the developer environment, including packages needed to run examples:
   ```bash
   pip install -e .[dev]
   ```
5. Make your changes to the code
6. Ensure your changes pass CI/CD routines:
   ```bash
   pytest
   ruff check aqua-blue/
   mypy aqua-blue/
   ```
7. Push your changes to your GitHub fork:
   ```bash
   git add path/to/changed/file1 path/to/changed/file
   git commit -m "Some short description of your changes"
   git push origin main
   ```
8. Open a pull request (instructions [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork))

You can also create your own branch if you can/want to! But I find forks easier for beginners 🙂