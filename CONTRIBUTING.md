# Community Contribution Guidelines

## Overview

### Contributing to core algorithms

In general, we are **NOT** accepting contributions to the core algorithms,
since we are a very small team with limited resources.

If you found any bugs, please report to us, and we will try to fix them.

### Adding code to `contrib` folder

You are welcome to contribute to the [`uisrnn/contrib`](uisrnn/contrib) folder.
For example, you can submit some scripts or tools, which you believe
others might find useful as well.
Please use
[`uisrnn/contrib/contrib_template.py`](uisrnn/contrib/contrib_template.py)
as an example.

#### Information to include

When you submit to [`uisrnn/contrib`](uisrnn/contrib),
please make sure you files contain these information:

* Your name.
* Your GitHub account.
* Your email address.
* (Optional) Your organization.
* A detailed docstring of what this file does, and how to use it.

Why do we need the contributor information:

* If people have questions on your code, we will redirect the questions to you.
* If people want to submit changes to the code that you authored, we will add
you as reviewer.

#### Unit tests

Your submitted code must include unit tests. See
[`tests/contrib/contrib_template_test.py`](tests/contrib/contrib_template_test.py)
 as an example

### Modifying code in `contrib` folder

If you want to modify the code in the `contrib` folder that was authored by
another contributor, please add the original author as the reviewer.
The author information can be found at the head of the file.

## Style

This repository follows Google's internal Python style.

Most importantly, we indent code blocks with *2 spaces*.

Suggested formatting tool: https://www.pylint.org/

Before you send a Pull Request, please use `run_pylint.sh` to check your style.
